class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            prec[ii]=rank_pos/rank[ii]                
            #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:   
                fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None

# init
self.aLRP_Loss = aLRPLoss()
self.SB_weight = 50
self.period = 3665
self.cls_LRP_hist = collections.deque(maxlen=self.period)
self.reg_LRP_hist = collections.deque(maxlen=self.period)
self.counter = 0

# __call__
def __call__(self, p, targets):  # predictions, targets
    lcls = torch.zeros(1, device=self.device)  # class loss
    lbox = torch.zeros(1, device=self.device)  # box loss
    lobj = torch.zeros(1, device=self.device)  # object loss
    tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

    # Losses
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
            pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

            # Regression
            pxy = pxy.sigmoid() * 2 - 0.5
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)

            # Classification
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                t[range(n), tcls[i]] = self.cp
                # lcls += self.BCEcls(pcls, t)  # BCE
                
                lbox_temp = 1.0 - iou
                losses_cls, rank, order = self.aLRP_Loss.apply(pcls.reshape(-1), t.reshape(-1), lbox_temp.detach())
                ordered_losses_bbox = lbox_temp[order.detach()].flip(dims=[0])
                losses_bbox = (torch.cumsum(ordered_losses_bbox,dim=0)/rank[order.detach()].detach().flip(dims=[0])).mean()
                
                self.cls_LRP_hist.append(float(losses_cls.item()))
                self.reg_LRP_hist.append(float(losses_bbox.item()))
                self.counter += 1
                
                if self.counter == self.period:
                    self.SB_weight = (np.mean(self.reg_LRP_hist)+np.mean(self.cls_LRP_hist))/np.mean(self.reg_LRP_hist)
                    self.cls_LRP_hist.clear()
                    self.reg_LRP_hist.clear()
                    self.counter=0
                
                lbox += losses_bbox * self.SB_weight  # iou loss
                lcls += losses_cls
            
            # Objectness
            iou = iou.detach().clamp(0).type(tobj.dtype)
            if self.sort_obj_iou:
                j = iou.argsort()
                b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
            if self.gr < 1:
                iou = (1.0 - self.gr) + self.gr * iou
            tobj[b, a, gj, gi] = iou  # iou ratio

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        obji = self.BCEobj(pi[..., 4], tobj)
        lobj += obji * self.balance[i]  # obj loss
        if self.autobalance:
            self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

    if self.autobalance:
        self.balance = [x / self.balance[self.ssi] for x in self.balance]
    lbox *= self.hyp['box']
    lobj *= self.hyp['obj']
    lcls *= self.hyp['cls']
    bs = tobj.shape[0]  # batch size

    return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()