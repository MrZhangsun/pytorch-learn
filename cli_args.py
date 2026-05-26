import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="",  choices=["cpu", "mps", "cuda"], help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_false", help="profile model speed")
    parser.add_argument("--line-profile", action="store_false", help="profile model speed layer by layer")
    parser.add_argument("--test", type=str, nargs="*",  default="sss", help="test all yolo*.yaml")
    opt = parser.parse_args()
    print(opt.test)
    print(opt.profile)