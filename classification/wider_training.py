def train(args, ctx):
    cl_weights = mx.nd.array(
        [1.0, 3.4595959, 18.472435, 3.3854823, 3.5971165, 1.1370194, 12.584616, 5.7822747, 10.827924, 1.7478157,
         8.8111115, 28.433332, 2.7568319, 18.020712])
    batch_ratios = nd.array(1 / cl_weights, ctx=ctx)

    class WeightedFocal(Loss):
        def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
            super(WeightedFocal, self).__init__(weight, batch_axis, **kwargs)
            self._from_sigmoid = from_sigmoid

        def hybrid_forward(self, F, pred, label, sample_weight=None):
            label = _reshape_like(F, label, pred)
            if not self._from_sigmoid:
                max_val = F.relu(-pred)
                loss = pred - pred * label + max_val + F.log(F.exp(-max_val) + F.exp(-pred - max_val))
            else:
                p = mx.nd.array(1 / (1 + nd.exp(-pred)), ctx=ctx)
                weights = nd.exp(label + (1 - label * 2) * batch_ratios)
                gamma = 2
                w_p, w_n = nd.power(1. - p, gamma), nd.power(p, gamma)
                loss = - (w_p * F.log(p + 1e-12) * label + w_n * F.log(1. - p + 1e-12) * (1. - label))
                loss *= weights
            return F.mean(loss, axis=self._batch_axis, exclude=True)


    class AttHistory(Loss):
        def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
            super(AttHistory, self).__init__(weight, batch_axis, **kwargs)
            self._from_sigmoid = from_sigmoid

        def hybrid_forward(self, F, pred, label, sample_weight=None):
            label = _reshape_like(F, label, pred)
            if not self._from_sigmoid:
                max_val = F.relu(-pred)
                loss = pred - pred * label + max_val + F.log(F.exp(-max_val) + F.exp(-pred - max_val))
            else:
                p = mx.nd.array(1 / (1 + nd.exp(-pred)), ctx=ctx)
                if epoch >= history_track and not args.test:
                    p_hist = prediction_history[:, batch_id * args.batch_size: (batch_id + 1) * args.batch_size, :]
                    p_std = (np.var(p_hist, axis=0) + (np.var(p_hist, axis=0)**2)/(p_hist.shape[0] - 1))**.5
                    std_weights = nd.array(1 + p_std, ctx=ctx)
                    loss = - std_weights * (F.log(p + 1e-12) * label + F.log(1. - p + 1e-12) * (1. - label))
                else:
                    loss = - (F.log(p + 1e-12) * label + F.log(1. - p + 1e-12) * (1. - label))
            return F.mean(loss, axis=self._batch_axis, exclude=True)