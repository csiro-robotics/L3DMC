from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


class DistilationLoss(torch.nn.Module):
    def __init__(self, temperature=2):
        self.temperature = temperature
        super(DistilationLoss, self).__init__()

    def forward(self, logits, targets, temperature=None):
        if temperature is not None:
            self.temperature = temperature
        soft_log_probs = torch.log_softmax(logits / self.temperature, dim=1)
        soft_targets = torch.softmax(targets / self.temperature, dim=1).detach()

        # return -torch.mean(torch.sum(soft_log_probs * soft_targets, dim=1, keepdim=False), dim=0, keepdim=False)
        return F.kl_div(soft_log_probs, soft_targets, reduction='batchmean')


class MCD(nn.Module):

	def __init__(self, s_dim, t_dim, feat_dim, c=1.0):
		super(MCD, self).__init__()
		self.embed_s_e = EmbedMLP(s_dim, feat_dim)
		self.embed_t_e = EmbedMLP(t_dim, feat_dim)

		self.embed_s_h = EmbedMLP(s_dim, feat_dim)
		self.embed_t_h = EmbedMLP(t_dim, feat_dim)

		self.c = c
		
		self.ball = geoopt.PoincareBall(c=self.c)

		self.criterion_h = SampleToSubspaceDistance()
		self.criterion_e = SampleToSubspaceDistance()

	def forward(self, feat_s_, feat_t_, weight_e=.5, weight_h=.5):
		feat_s = feat_s_ + torch.zeros_like(feat_s_).normal_(0, 0.0001)
		feat_t = feat_t_ + torch.zeros_like(feat_t_).normal_(0, 0.0001)
		
		# Euclidean Space

		# With Projecting input feature onto new projection space
		feat_s_e = self.embed_s_e(feat_s)
		feat_t_e = self.embed_t_e(feat_t)

		# Without projection
		# loss_e = self.criterion_e(feat_s, feat_t)
		# With Projection
		loss_e = self.criterion_e(feat_s_e, feat_t_e)


		# Hyperbolic Space

		# Without Projecting input feature into new projection space
		# feat_s_h = self.ball.projx(feat_s)
		# feat_t_h = self.ball.projx(feat_t)
		
		# With Projecting input feature into new projection space
		feat_s_h = self.ball.projx(self.embed_s_h(feat_s))
		feat_t_h = self.ball.projx(self.embed_t_h(feat_t))

		# loss_h = self.criterion_h(self.ball_s.logmap0(feat_s_h, c=self.c), self.ball_t.logmap0(feat_t_h, c=self.c))
		loss_h = self.criterion_h(self.ball.logmap0(feat_s_h), self.ball.logmap0(feat_t_h))
		
		# Weighted sum of losses
		loss = weight_e*loss_e + weight_h*loss_h

		return loss


class EmbedLinear(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(EmbedLinear, self).__init__()
		self.linear = nn.Linear(in_dim, out_dim)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		x = F.normalize(x, p=2, dim=1)

		return x


class EmbedMLP(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(EmbedMLP, self).__init__()
		self.mlp = nn.Sequential(
		nn.Linear(in_dim, out_dim),
		nn.ReLU(inplace=True),
		nn.Linear(out_dim, out_dim))
		
	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.mlp(x)
		
		x = F.normalize(x, p=2, dim=1)

		return x


class SampleToSubspaceDistance(nn.Module):

	def __init__(self, method='sample2subspace'):
		super(SampleToSubspaceDistance, self).__init__()
		self.kernel = lambda anchor_feature, contrast_feature: torch.exp(-(torch.cdist(anchor_feature, contrast_feature)**2)*.5)
		self.method = method

	def forward(self, feat_s, feat_t):
		bs = feat_s.size(0)
		covar_matrix_s = self.kernel(feat_s, feat_s).to_dense()
		covar_matrix_t = self.kernel(feat_t, feat_t).to_dense()
  
		covar_matrix_st = self.kernel(feat_s, feat_t).to_dense()
		covar_matrix_ts = self.kernel(feat_t, feat_s).to_dense()
		
		right_term = torch.mm(covar_matrix_st, torch.mm(torch.inverse(covar_matrix_t), covar_matrix_ts))
		loss = (covar_matrix_s.diag() - right_term.diag()).mean()

		return loss
