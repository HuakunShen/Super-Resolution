# Good Results


## DIV2KCustom

### Observations
- For Unets, UNetNoTop is the worst, UNetD4 the best

### Results
- **FSRCNN_Original** (SR OK, but many noise)
- **FSRCNN-100-300** (SR a little bit, noise exists, not very good)
- **FSRCNN-100-300-original** (Not so good)
- **resnet_pretrainedDS_new** (very bad, blur)
- **ResNetPretrainedDS** (very bad, blur)
- **srcnn_100_300_perceptual_loss** (OK, improved a little)
- **srcnn_100_300_perceptual_loss_2** (seems good)
- **SRCNN-50-300-100iter-scheduler** (bad)
- **SRCNN-100-300** (Good)
- **SRCNN-100-300-100iter-scheduler** (Bad, become grayscale, scheduler doesn't always improve)
- **SRCNN-100-300-50iter** (ok)
- **unetd4_100_300_perceptual_loss** (pretty good results with a little color flaw)
- **unetNoTop** (very bad, very blur)
- **unetsr_100_300_perceptual_loss_w_seed** (good)
- **UNetSR-100-300-150iter** (ok)
- **UNetSR-100-300-50iter-scheduler** (not good, color sometimes wrong)



## TEXT

### Observations:
- Unet Converge Faster

### Results
- **srcnn-batch-norm-blur-3-gamma-0.9-MSE** (flaw)
- **srcnn-blur-3** (OK)
- **srcnn-blur-3-gamma-0.9-MSE** (Flaw)
- **srcnn-blur-3-gamma-0.9-perceptual** (Bad)
- **unetd4-blur-3-MSE** (OK with flaw)
- **unetsr-blur-3** (some are good, many flaws, unetsr not as good as unetd4)
- **unetsr-blur-3-MSE** (very bad, very blur)
- **unetd4-blur-5-MSE** (OK, a little bit flaw)
- **unetd4-blur-5-perceptual** (Good, test images have some bad results)
- **unetd4-blur-7-perceptual** (good)
- **unetd4-blur-7-MSE** (OK, There are still many images with flaws, especially images with pure white background)
