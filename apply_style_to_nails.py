import argparse
import cv2
import numpy as np
import os

from nails_segmantation_inference import segment_nails


def apply_style(img_path, style_path, model, preprocessing, mixing_level):
    mask, alpha_channel = segment_nails(img_path, model, preprocessing)
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    style = cv2.cvtColor(cv2.imread(style_path), cv2.COLOR_BGR2RGB)
    im_h, im_w = image.shape[:2]
    style = cv2.resize(style, (im_w, im_h))

    image_no_nails = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    image_only_nails = cv2.bitwise_and(image, image, mask=mask)
    style_nails = cv2.bitwise_and(style, style, mask=mask)
    alpha_channel_3 = np.repeat(alpha_channel[:,:,np.newaxis], 3, axis=2)
    alpha_channel_3 *= mixing_level
    styled_mixed_nails = np.float32(np.float32(image_only_nails) * (1 - alpha_channel_3)) + np.float32(style_nails) * alpha_channel_3
    image_with_styled_nails = image_no_nails + np.uint8(styled_mixed_nails)
    
    # image_name = os.path.splitext(os.path.basename(img_path))[0]
    # cv2.imwrite(f'{image_name}_result_{int(100 * mixing_level)}.jpg', cv2.cvtColor(image_with_styled_nails, cv2.COLOR_RGB2BGR))
    # image_nails_masked = image.copy()
    # image_nails_masked[:,:,0] += cv2.bitwise_and(image_nails_masked[:,:,0], mask) * 255
    # cv2.imwrite(f'{image_name}_mask.jpg', cv2.cvtColor(image_nails_masked, cv2.COLOR_RGB2BGR))
    return image_with_styled_nails


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
    
    apply_style(args.img_path, args.style_path, args.model, args.preprocessing, args.mixing_level)
