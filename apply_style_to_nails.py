import argparse
import cv2
import numpy as np

from nails_segmantation_inference import segment_nails

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", required = True, help = 'Path to image for processing')
    ap.add_argument("-s", "--style_path", required = True, help = 'Path to style to apply')
    ap.add_argument("-m", "--model", 
                    default = "nails_segmantation_inference/best_model_80_v1.pth", 
                    help = 'Path to pth model')
    ap.add_argument("-p", "--preprocessing", 
                    default = "nails_segmantation_inference/preprocessing_fn_80_v1.pkl",
                    help = 'Path to preprocessing pkl')
    ap.add_argument("--mixing_level", default=0.5, type=float, help = 'Style mixing level')
    args = ap.parse_args()
    mask = segment_nails(args.img_path, args.model, args.preprocessing)
    
    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    style = cv2.cvtColor(cv2.imread(args.style_path), cv2.COLOR_BGR2RGB)
    im_h, im_w = image.shape[:2]
    style = cv2.resize(style, (im_w, im_h))
    
    image_no_nails = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    image_only_nails = cv2.bitwise_and(image, image, mask=mask)
    style_nails = cv2.bitwise_and(style, style, mask=mask)
    styled_mixed_nails = np.float32(np.float32(image_only_nails) * (1 - args.mixing_level) + style_nails * args.mixing_level)
    image_with_styled_nails = image_no_nails + np.uint8(styled_mixed_nails)
    cv2.imwrite(f'image_with_style_{int(100 * args.mixing_level)}.jpg', cv2.cvtColor(image_with_styled_nails, cv2.COLOR_RGB2BGR))