# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import logging
import os
import os.path as osp
import sys

from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity, ModelOptimizationType, ModelPrecision, ModelStatus, OptimizationMethod
from ote_sdk.entities.model_template import TargetDevice, parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter

from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils import get_task_class
from mmdet.apis.ote.extension.datasets.data_utils import load_dataset_items_coco_format
from sc_sdk.entities.datasets import NullDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_model_weights(path):
    with open(path, 'rb') as read_file:
        return read_file.read()



def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--data-dir', default='/media/cluster_fs/datasets/object_detection/pcd-coco/')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--load-weights', required=True, help='Load only weights from previously saved checkpoint')
    parser.add_argument('--save-to', default='IR')

    return parser.parse_args()


def get_gold_accuracy(args):
    labels_list = []
    items = load_dataset_items_coco_format(
        ann_file_path=osp.join(args.data_dir, 'annotations/instances_train.json'),
        data_root_dir=osp.join(args.data_dir, 'images/train/'),
        subset=Subset.TRAINING,
        labels_list=labels_list)
    items.extend(load_dataset_items_coco_format(
        ann_file_path=osp.join(args.data_dir, 'annotations/instances_val.json'),
        data_root_dir=osp.join(args.data_dir, 'images/val/'),
        subset=Subset.VALIDATION,
        labels_list=labels_list))
    items.extend(load_dataset_items_coco_format(
        ann_file_path=osp.join(args.data_dir, 'annotations/instances_val.json'),
        data_root_dir=osp.join(args.data_dir, 'images/val/'),
        subset=Subset.TESTING,
        labels_list=labels_list))
    dataset = DatasetEntity(items=items)

    labels_schema = LabelSchemaEntity.from_labels(labels_list)

    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(args.template_file_path)

    hyper_parameters = model_template.hyper_parameters.data
    set_values_as_default(hyper_parameters)

    logger.info('Setup environment')
    params = create(hyper_parameters)
    logger.info('Set hyperparameters')
    params.learning_parameters.num_iters = 1
    environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                  model_template=model_template)

    model_bytes = load_model_weights(args.load_weights)
    input_model = ModelEntity(configuration=environment.get_model_configuration(),
                                model_adapters={'weights.pth': ModelAdapter(model_bytes)},
                                train_dataset=NullDataset())

    environment.model = input_model


    logger.info('Create NNCF Task')
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info('Get predictions on the validation set')
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))

    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY)

    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info('Estimate quality on validation set')
    task.evaluate(resultset)
    logger.info(str(resultset.performance))
    return resultset.performance

def main(args):
    fp32_metric = get_gold_accuracy(args)

    logger.info('Initialize dataset')
    labels_list = []
    items = load_dataset_items_coco_format(
        ann_file_path=osp.join(args.data_dir, 'annotations/instances_train.json'),
        data_root_dir=osp.join(args.data_dir, 'images/train/'),
        subset=Subset.TRAINING,
        labels_list=labels_list)
    items.extend(load_dataset_items_coco_format(
        ann_file_path=osp.join(args.data_dir, 'annotations/instances_val.json'),
        data_root_dir=osp.join(args.data_dir, 'images/val/'),
        subset=Subset.VALIDATION,
        labels_list=labels_list))
    items.extend(load_dataset_items_coco_format(
        ann_file_path=osp.join(args.data_dir, 'annotations/instances_val.json'),
        data_root_dir=osp.join(args.data_dir, 'images/val/'),
        subset=Subset.TESTING,
        labels_list=labels_list))
    dataset = DatasetEntity(items=items)

    labels_schema = LabelSchemaEntity.from_labels(labels_list)

    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(args.template_file_path)

    hyper_parameters = model_template.hyper_parameters.data
    set_values_as_default(hyper_parameters)

    logger.info('Setup environment')
    params = create(hyper_parameters)
    logger.info('Set hyperparameters')
    params.learning_parameters.num_iters = 1
    environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema, model_template=model_template)

    model_bytes = load_model_weights(args.load_weights)
    input_model = ModelEntity(configuration=environment.get_model_configuration(),
                                model_adapters={'weights.pth': ModelAdapter(model_bytes)},
                                train_dataset=NullDataset())

    environment.model = input_model


    logger.info('Create NNCF Task')
    task_impl_path = model_template.entrypoints.nncf
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info('Optimize model')
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY)
    optimize_parameters = OptimizationParameters()
    task.optimize(OptimizationType.NNCF, dataset, output_model, optimize_parameters)

    logger.info('Get predictions on the validation set')
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))

    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info('Estimate quality of NNCF model on validation set:')
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    logger.info('Estimate quality of FP32 model on validation set:')
    logger.info(str(fp32_metric))

    if args.export:
        logger.info('Export model')
        exported_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        task.export(ExportType.OPENVINO, exported_model)

        nncf_export_dir = osp.join(args.save_to, "nncf")
        os.makedirs(nncf_export_dir, exist_ok=True)
        with open(osp.join(nncf_export_dir, "openvino.bin"), "wb") as f:
            f.write(exported_model.get_data("openvino.bin"))
        with open(osp.join(nncf_export_dir, "openvino.xml"), "wb") as f:
            f.write(exported_model.get_data("openvino.xml"))

        # logger.info('Create OpenVINO Task')
        # environment.model = exported_model
        # openvino_task_impl_path = model_template.entrypoints.openvino
        # openvino_task_cls = get_task_class(openvino_task_impl_path)
        # openvino_task = openvino_task_cls(environment)
        #
        # logger.info('Get predictions on the validation set')
        # predicted_validation_dataset = openvino_task.infer(
        #     validation_dataset.with_empty_annotations(),
        #     InferenceParameters(is_evaluation=True))
        # resultset = ResultSetEntity(
        #     model=output_model,
        #     ground_truth_dataset=validation_dataset,
        #     prediction_dataset=predicted_validation_dataset,
        # )
        # logger.info('Estimate quality on validation set')
        # performance = openvino_task.evaluate(resultset)
        # logger.info(str(performance))

if __name__ == '__main__':
    sys.exit(main(parse_args()) or 0)
