# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models.moco.moco_wrapper import MoCoWrapper


class PCLWrapper(MoCoWrapper):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def forward_loss(self, im_q, im_k, psedo_labels, cluster_centers, density):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        contrastive_loss = F.cross_entropy(logits, labels)

        cls_loss = 0.
        for num_cluster in psedo_labels:
            # logits = q.mm(cluster_centers[num_cluster].T) / self.cluster_temp
            # cls_loss += F.cross_entropy(logits, psedo_labels[num_cluster])
            # get positive prototypes
            pos_proto_id = psedo_labels[num_cluster]
            prototypes = cluster_centers[num_cluster]

            pos_prototypes = prototypes[pos_proto_id]

            # sample negative prototypes
            all_proto_id = [i for i in range(num_cluster)]
            neg_proto_id = set(all_proto_id) - set(pos_proto_id.cpu().numpy().tolist())

            # print(len(neg_proto_id), neg_proto_id, self.K)
            # neg_proto_id = random.sample(neg_proto_id, self.K)  # sample r negative prototypes
            neg_proto_id = random.choices(list(neg_proto_id), k=self.K)
            neg_proto_id = torch.LongTensor(neg_proto_id).cuda()

            neg_prototypes = prototypes[neg_proto_id]

            # proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)
            #
            # # compute prototypical logits
            # logits_proto = torch.mm(q, proto_selected.t())
            #
            # # targets for prototype assignment
            # labels_proto = torch.linspace(0, q.size(0) - 1, steps=q.size(0)).long().cuda()
            #
            # # scaling temperatures for the selected prototypes
            # temp_proto = density[num_cluster][torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).cuda()], dim=0)]

            # positive logits: Nx1
            logits_proto_pos = torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1)
            # negative logits: NxK
            logits_proto_neg = torch.einsum('nc,kc->nk', [q, neg_prototypes])

            # logits_proto: Nx(1+K)
            logits_proto = torch.cat([logits_proto_pos, logits_proto_neg], dim=1)

            labels_proto = torch.zeros(logits_proto.shape[0], dtype=torch.long).cuda()

            # scaling temperatures for the selected prototypes
            temp_proto = torch.cat([pos_proto_id.unsqueeze(1),
                                    neg_proto_id.unsqueeze(0).repeat(pos_proto_id.size(0), 1)], dim=1)
            temp_proto = density[num_cluster][temp_proto]

            logits_proto /= temp_proto

            cls_loss += F.cross_entropy(logits_proto, labels_proto)

        cls_loss /= len(psedo_labels)

        return contrastive_loss, cls_loss, k

    def forward(self, im_q, im_k, psedo_labels: dict, cluster_centers: dict, density: dict):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            # for num_cluster in psedo_labels:
            #     psedo_labels[num_cluster] = psedo_labels[num_cluster].repeat(2)
            # contrastive_loss, cls_loss, k = self.forward_loss(torch.cat([im_q, im_k], dim=0),
            #                                                   torch.cat([im_k, im_q], dim=0),
            #                                                   psedo_labels, cluster_centers, density)
            contrastive_loss1, cls_loss1, k1 = self.forward_loss(im_q, im_k, psedo_labels, cluster_centers, density)
            contrastive_loss2, cls_loss2, k2 = self.forward_loss(im_k, im_q, psedo_labels, cluster_centers, density)
            contrastive_loss = contrastive_loss1 + contrastive_loss2
            cls_loss = cls_loss1 + cls_loss2
            k = torch.cat([k1, k2], dim=0)

        else:  # asymmetric loss
            contrastive_loss, cls_loss, k = self.forward_loss(im_q, im_k, psedo_labels, cluster_centers, density)

        self._dequeue_and_enqueue(k)

        return contrastive_loss, cls_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
