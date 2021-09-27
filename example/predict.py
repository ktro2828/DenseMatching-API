#!/usr/bin/env python3

import argparse
import base64
import requests

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://0.0.0.0:8000/predict", help="target url")
    parser.add_argument("--host", type=str, help="[Optional] target host")
    parser.add_argument("--port", type=str, help="[Optional] target port")
    parser.add_argument("-q", "--query", type=str, required=True)
    parser.add_argument("-r", "--reference", type=str, required=True)
    parser.add_argument("--size", nargs="+", default=[752, 480])
    parser.add_argument("--flipping_condition", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    headers = {"accept": "application/json"}
    params = {"size": [752, 480], "flipping_condition": args.flipping_condition}
    files = {"query": open(args.query, "rb"), "reference": open(args.reference, "rb")}
    if args.host is not None and args.port is not None:
        url = "http://" + args.host + ":" + args.port + "/predict"
    else:
        url = args.url
    response = requests.post(url=url, headers=headers, files=files, params=params)

    try:
        ret = response.json()
        b64_warped_img = ret["warped_image"]
        b64_est_flow = ret["estimated_flow"]
        b64_warped_conf_map = ret["warped_confidence_map"]

        warped_str = base64.b64decode(b64_warped_img)
        warped_nparr = np.fromstring(warped_str, dtype=np.uint8)
        warped_img = cv2.imdecode(warped_nparr, cv2.IMREAD_COLOR)
        h, w = warped_img.shape[:2]
        ref_img = cv2.imread(args.reference)
        blend_img = cv2.addWeighted(warped_img, 0.5, ref_img, 0.5, 0)
        cv2.imshow("Warped image", warped_img)
        cv2.imshow("Blend image", blend_img)
        cv2.waitKey(0)

        est_flow_str = base64.b64decode(b64_est_flow)
        est_flow = np.fromstring(est_flow_str, dtype=np.float32)
        est_flow = est_flow.reshape(h, w, 2)
        print("Estimated flow: ", est_flow)
        if b64_warped_conf_map is not None:
            warped_conf_map_ = base64.b64decode(b64_warped_conf_map)
            warped_conf_map_nparr = np.fromstring(warped_conf_map_str, dtype=np.uint8)
            warped_conf_map = cv2.imdecode(warped_conf_map_nparr, cv2.IMREAD_COLOR)
            cv2.imshow("Confidence map", warped_conf_map)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(response.status_code, response.reason)
        print(e)


if __name__ == "__main__":
    main()
