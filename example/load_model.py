#!/usr/bin/env python3

import argparse
import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://0.0.0.0:8000/model")
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=str)
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-p", "--pre_trained_model_type", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.host is not None and args.port is not None:
        url = "http://" + args.host + ":" + args.port + "/model"
    else:
        url = args.url
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    params = {"name": args.name, "pre_trained_model_type": args.pre_trained_model_type}
    response = requests.post(url=url, headers=headers, params=params)

    print(response.status_code, response.reason)
    print(response)
    print(response.json())


if __name__ == "__main__":
    main()
