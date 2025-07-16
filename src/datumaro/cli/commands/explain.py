# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.util.image import is_image, load_image, save_image
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import load_project, parse_full_revpath

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Run Explainable AI algorithm",
        description=f"""
        {deprecated} Runs an explainable AI algorithm for a model.|n
        |n
        This tool is supposed to help an AI developer to debug
        a model and a dataset. Basically, it executes inference and
        tries to find problems in the trained model - determine decision
        boundaries and belief intervals for the classifier.|n
        |n
        Currently, the only available algorithm is
        RISE (https://arxiv.org/pdf/1806.07421.pdf), which runs
        inference and then re-runs a model multiple times
        on each image to produce a heatmap of activations for
        each output of the first inference. As a result, we obtain
        few heatmaps, which shows, how image pixels affected
        the inference result. This algorithm doesn't require any special
        information about the model, but it requires the model to
        return all the outputs and confidences. Check the User Manual
        for usage examples.|n
        Supported scenarios:|n
        - RISE for classification|n
        - RISE for Object Detection|n
        |n
        This command has the following syntax:|n
        |s|s%(prog)s <image path or revpath>|n
        |n
        <image path> - a path to the file.|n
        <revpath> - either a dataset path or a revision path. The full
        syntax is:|n
        - Dataset paths:|n
        |s|s- <dataset path>[ :<format> ]|n
        - Revision paths:|n
        |s|s- <project path> [ @<rev> ] [ :<target> ]|n
        |s|s- <rev> [ :<target> ]|n
        |s|s- <target>|n
        Parts can be enclosed in quotes.|n
        |n
        The current project (-p/--project) is used as a context for plugins
        and models. It is used when there is a dataset path in target.
        When not specified, the current project's working tree is used.|n
        |n
        Examples:|n
        - Run RISE on an image, display results:|n
        |s|s%(prog)s path/to/image.jpg -m mymodel rise --max-samples 50|n
        |n
        - Run RISE on a source revision:|n
        |s|s%(prog)s HEAD~1:source-1 -m model rise
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Inference target - image, revpath (default: project)",
    )
    parser.add_argument("-m", "--model", required=True, help="Model to use for inference")
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="save_dir",
        default=None,
        help="Directory to save output (default: display only)",
    )

    method_sp = parser.add_subparsers(dest="algorithm")

    rise_parser = method_sp.add_parser(
        "rise",
        description="""
        RISE: Randomized Input Sampling for
        Explanation of Black-box Models algorithm|n
        |n
        See explanations at: https://arxiv.org/pdf/1806.07421.pdf
        """,
        formatter_class=MultilineFormatter,
    )
    rise_parser.add_argument(
        "-s",
        "--max-samples",
        default=100,
        type=int,
        help="Number of algorithm iterations (default: mask size ^ 2)",
    )
    rise_parser.add_argument(
        "--mw",
        "--mask-width",
        dest="mask_width",
        default=7,
        type=int,
        help="Mask width (default: %(default)s)",
    )
    rise_parser.add_argument(
        "--mh",
        "--mask-height",
        dest="mask_height",
        default=7,
        type=int,
        help="Mask height (default: %(default)s)",
    )
    rise_parser.add_argument(
        "--prob",
        default=0.5,
        type=float,
        help="Mask pixel inclusion probability (default: %(default)s)",
    )
    rise_parser.add_argument(
        "--iou",
        "--iou-thresh",
        dest="iou_thresh",
        default=0.9,
        type=float,
        help="IoU match threshold for detections (default: %(default)s)",
    )
    rise_parser.add_argument(
        "--nms",
        "--nms-iou-thresh",
        dest="nms_iou_thresh",
        default=0.0,
        type=float,
        help="IoU match threshold in Non-maxima suppression (default: no NMS)",
    )
    rise_parser.add_argument(
        "--conf",
        "--det-conf-thresh",
        dest="det_conf_thresh",
        default=0.0,
        type=float,
        help="Confidence threshold for detections (default: include all)",
    )
    rise_parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Inference batch size (default: %(default)s)",
    )
    rise_parser.add_argument(
        "--display", action="store_true", help="Visualize results during computations"
    )

    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=explain_command)

    return parser


def get_sensitive_args():
    return {
        explain_command: [
            "target",
            "model",
            "save_dir",
            "project_dir",
        ],
    }


@scoped
def explain_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    import cv2
    from matplotlib import cm

    project = scope_add(load_project(args.project_dir))

    if args.model not in project.models:
        raise CliException(
            f"{args.model} has not been registered in this project. "
            f"This is a list of registered model names: {list(project.models.keys())}"
        )

    model = project.make_model(args.model)

    if str(args.algorithm).lower() != "rise":
        raise CliException("Explanation algorithm except for rise is not yet implemented.")

    from datumaro.components.algorithms.rise import RISE

    rise = RISE(
        model,
        num_masks=args.max_samples,
        mask_size=args.mask_width,
        prob=args.prob,
        batch_size=args.batch_size,
    )

    if args.target and is_image(args.target):
        image_path = args.target
        image = load_image(image_path)

        log.info("Running inference explanation for '%s'" % image_path)
        heatmap_iter = rise.apply(image, progressive=args.display)

        image = image / 255.0
        file_name = osp.splitext(osp.basename(image_path))[0]
        if args.display:
            for i, heatmaps in enumerate(heatmap_iter):
                for j, heatmap in enumerate(heatmaps):
                    hm_painted = cm.jet(heatmap)[:, :, 2::-1]
                    disp = (image + hm_painted) / 2
                    cv2.imshow("heatmap-%s" % j, hm_painted)
                    cv2.imshow(file_name + "-heatmap-%s" % j, disp)
                cv2.waitKey(10)
                print("Iter", i, "of", args.max_samples, end="\r")
        else:
            heatmaps = next(heatmap_iter)

        if args.save_dir is not None:
            log.info("Saving inference heatmaps at '%s'" % args.save_dir)
            os.makedirs(args.save_dir, exist_ok=True)

            for j, heatmap in enumerate(heatmaps):
                save_path = osp.join(args.save_dir, file_name + "-heatmap-%s.png" % j)
                save_image(save_path, heatmap * 255.0)
        else:
            for j, heatmap in enumerate(heatmaps):
                disp = (image + cm.jet(heatmap)[:, :, 2::-1]) / 2
                cv2.imshow(file_name + "-heatmap-%s" % j, disp)
            cv2.waitKey(0)

    else:
        dataset, target_project = parse_full_revpath(args.target or "project", project)
        if target_project:
            scope_add(target_project)

        log.info("Running inference explanation for '%s'" % args.target)

        for item in dataset:
            image = item.image.data
            if image is None:
                log.warning("Item %s does not have image data. Skipping.", item.id)
                continue

            heatmap_iter = rise.apply(image)

            image = image / 255.0
            heatmaps = next(heatmap_iter)

            if args.save_dir is not None:
                log.info("Saving inference heatmaps to '%s'" % args.save_dir)
                os.makedirs(args.save_dir, exist_ok=True)

                for j, heatmap in enumerate(heatmaps):
                    save_image(
                        osp.join(args.save_dir, item.id + "-heatmap-%s.png" % j),
                        heatmap * 255.0,
                        create_dir=True,
                    )

            if not args.save_dir or args.display:
                for j, heatmap in enumerate(heatmaps):
                    disp = (image + cm.jet(heatmap)[:, :, 2::-1]) / 2
                    cv2.imshow(item.id + "-heatmap-%s" % j, disp)
                cv2.waitKey(0)

    return 0
