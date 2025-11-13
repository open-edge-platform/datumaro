# Copyright (C) 2020-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Tuple

from attrs import define, field, validators

from datumaro.util.attrs_util import has_length, not_empty


class ImmutableObjectError(Exception):
    def __str__(self):
        return "Cannot set value of immutable object"


class DatumaroError(Exception):
    pass


class VcsError(DatumaroError):
    pass


class ReadonlyDatasetError(VcsError):
    def __str__(self):
        return "Can't update a read-only dataset"


class ReadonlyProjectError(VcsError):
    def __str__(self):
        return "Can't change a read-only project"


@define(auto_exc=False)
class UnknownRefError(VcsError):
    ref = field()

    def __str__(self):
        return f"Can't parse ref '{self.ref}'"


class MissingObjectError(VcsError):
    pass


class MismatchingObjectError(VcsError):
    pass


@define(auto_exc=False)
class UnsavedChangesError(VcsError):
    paths = field()

    def __str__(self):
        return "There are some uncommitted changes: %s" % ", ".join(self.paths)


class ForeignChangesError(VcsError):
    pass


class EmptyCommitError(VcsError):
    pass


class PathOutsideSourceError(VcsError):
    pass


class SourceUrlInsideProjectError(VcsError):
    def __str__(self):
        return "Source URL cannot point inside the project"


class UnexpectedUrlError(VcsError):
    pass


class MissingSourceHashError(VcsError):
    pass


class PipelineError(DatumaroError):
    pass


class InvalidPipelineError(PipelineError):
    pass


class EmptyPipelineError(InvalidPipelineError):
    pass


class MultiplePipelineHeadsError(InvalidPipelineError):
    pass


class MissingPipelineHeadError(InvalidPipelineError):
    pass


class InvalidStageError(InvalidPipelineError):
    pass


class UnknownStageError(InvalidStageError):
    pass


class MigrationError(DatumaroError):
    pass


class OldProjectError(DatumaroError):
    def __str__(self):
        return """
            The project you're trying to load was
            created by the old Datumaro version. Try to migrate the
            project with 'datum project migrate' and then reload.
            """


class NotAvailableError(DatumaroError):
    pass


@define(auto_exc=False)
class ProjectNotFoundError(DatumaroError):
    path = field()

    def __str__(self):
        return f"Can't find project at '{self.path}'"


@define(auto_exc=False)
class ProjectAlreadyExists(DatumaroError):
    path = field()

    def __str__(self):
        return f"Can't create project: a project already exists at '{self.path}'"


@define(auto_exc=False)
class VcsAlreadyExists(DatumaroError):
    path = field()

    def __str__(self):
        return f"Can't create project: a version control system already exists at '{self.path}'"


@define(auto_exc=False)
class UnknownSourceError(DatumaroError):
    name = field()

    def __str__(self):
        return f"Unknown source '{self.name}'"


@define(auto_exc=False)
class UnknownTargetError(DatumaroError):
    name = field()

    def __str__(self):
        return f"Unknown target '{self.name}'"


@define(auto_exc=False)
class UnknownFormatError(DatumaroError):
    format = field()

    def __str__(self):
        return (
            f"Unknown source format '{self.format}'. To make it "
            "available, add the corresponding Extractor implementation "
            "to the environment"
        )


@define(auto_exc=False)
class SourceExistsError(DatumaroError):
    name = field()

    def __str__(self):
        return f"Source '{self.name}' already exists"


class DatasetExportError(DatumaroError):
    pass


@define(auto_exc=False)
class ItemExportError(DatasetExportError):
    """
    Represents additional item error info. The error itself is supposed to be
    in the `__cause__` member.
    """

    item_id: Tuple[str, str]

    def __str__(self):
        return "Failed to export item %s" % (self.item_id,)


class AnnotationExportError(ItemExportError):
    pass


class DatasetImportError(DatumaroError):
    pass


class InvalidAnnotationError(DatasetImportError):
    """
    A basic dataset parsing error. Should include the problem description in
    the message.
    """


@define(auto_exc=False)
class InvalidFieldError(InvalidAnnotationError):
    name: str = field(validator=[validators.instance_of(str), not_empty])
    """Field name"""

    def __str__(self) -> str:
        return f"Invalid annotation field '{self.name}' value"


@define(auto_exc=False)
class InvalidFieldTypeError(InvalidFieldError):
    actual: str = field(validator=[validators.instance_of(str), not_empty])
    """Actual type of the field"""

    expected: Tuple[str] = field(validator=[validators.instance_of(tuple), not_empty])
    """The list of expected types of the field"""

    def __str__(self) -> str:
        if len(self.expected) == 1:
            expected = self.expected[0]
        else:
            expected = "one of " + ", ".join(self.expected)
        return f"Invalid annotation field '{self.name}' type '{self.actual}'. Expected {expected}"


@define(auto_exc=False)
class MissingFieldError(InvalidFieldError):
    def __str__(self) -> str:
        return f"Missing annotation field '{self.name}'"


@define(auto_exc=False)
class UndeclaredLabelError(InvalidAnnotationError):
    id: str = field(validator=validators.instance_of(str))
    """Index or name"""

    def __str__(self) -> str:
        return f"Undeclared label '{self.id}'"


@define(auto_exc=False)
class ItemImportError(DatasetImportError):
    """
    Wraps a dataset parsing error and provides additional error context info.
    The error itself is supposed to be in the `__cause__` member.
    """

    item_id: Tuple[Optional[str], Optional[str]] = field(validator=[validators.instance_of(tuple), has_length(2)])
    """
    (id, subset) of the item with problem.
    If id or subset cannot be reported, such field is set to None.
    """

    def __str__(self):
        return f"Failed to import item {self.item_id}: {self.__cause__}"


class AnnotationImportError(ItemImportError):
    def __str__(self):
        return f"Failed to import item {self.item_id} annotation: {self.__cause__}"


@define(auto_exc=False)
class DatasetNotFoundError(DatasetImportError):
    path: str = field()
    format: str = field()
    template: str = field(default="Failed to find dataset '{format}' at '{path}'")

    def __str__(self):
        return self.template.format(path=self.path, format=self.format)


@define(auto_exc=False)
class MultipleFormatsMatchError(DatasetImportError):
    formats = field()

    def __str__(self):
        return "Failed to detect dataset format automatically: data matches more than one format: %s" % ", ".join(
            self.formats
        )


class NoMatchingFormatsError(DatasetImportError):
    def __str__(self):
        return "Failed to detect dataset format automatically: no matching formats found"


class DatasetError(DatumaroError):
    pass


class AnnotationTypeError(DatumaroError):
    pass


class MediaTypeError(DatumaroError):
    pass


class MediaShapeError(DatumaroError):
    pass


class DatasetInfosRedefinedError(DatasetError):
    def __str__(self):
        return "Infos can only be set once for a dataset"


class CategoriesRedefinedError(DatasetError):
    def __str__(self):
        return "Categories can only be set once for a dataset"


@define(auto_exc=False)
class RepeatedItemError(DatasetError):
    item_id = field()

    def __str__(self):
        return f"Item {self.item_id} is repeated in the source sequence."


@define(auto_exc=False)
class PathSeparatorInSubsetNameError(DatasetError):
    subset: str = field()

    def __str__(self):
        return f"Failed to export the subset '{self.subset}': subset name contains path separator(s)."


class DatasetQualityError(DatasetError):
    pass


@define(auto_exc=False)
class AnnotationsTooCloseError(DatasetQualityError):
    item_id = field()
    a = field()
    b = field()
    distance = field()

    def __str__(self):
        return "Item %s: annotations are too close: %s, %s, distance = %s" % (
            self.item_id,
            self.a,
            self.b,
            self.distance,
        )


@define(auto_exc=False)
class WrongGroupError(DatasetQualityError):
    item_id = field()
    found = field(converter=set)
    expected = field(converter=set)
    group = field(converter=list)

    def __str__(self):
        return "Item %s: annotation group has wrong labels: found %s, expected %s, group %s" % (
            self.item_id,
            self.found,
            self.expected,
            self.group,
        )


@define(auto_exc=False, init=False)
class DatasetMergeError(DatasetError):
    sources = field(converter=set, factory=set, kw_only=True)

    def _my__init__(self, msg=None, *, sources=None):
        super().__init__(msg)
        self.__attrs_init__(sources=sources or set())


# Pylint will raise false positive warnings for derived classes,
# when __init__ is defined directly
setattr(DatasetMergeError, "__init__", DatasetMergeError._my__init__)


@define(auto_exc=False)
class MismatchingImageInfoError(DatasetMergeError):
    item_id: Tuple[str, str]
    a: Tuple[int, int]
    b: Tuple[int, int]

    def __str__(self):
        return "Item %s: mismatching image size info: %s vs %s" % (self.item_id, self.a, self.b)


@define(auto_exc=False)
class MismatchingMediaPathError(DatasetMergeError):
    item_id: Tuple[str, str]
    a: str
    b: str

    def __str__(self):
        return "Item %s: mismatching media path info: %s vs %s" % (self.item_id, self.a, self.b)


@define(auto_exc=False)
class MismatchingMediaError(DatasetMergeError):
    item_id: Tuple[str, str]
    a: Any
    b: Any

    def __str__(self):
        return "Item %s: mismatching media info: %s vs %s" % (self.item_id, self.a, self.b)


@define(auto_exc=False)
class MismatchingAttributesError(DatasetMergeError):
    item_id: Tuple[str, str]
    key: str
    a: Any
    b: Any

    def __str__(self):
        return "Item %s: mismatching image attribute %s: %s vs %s" % (
            self.item_id,
            self.key,
            self.a,
            self.b,
        )


class ConflictingCategoriesError(DatasetMergeError):
    pass


@define(auto_exc=False)
class NoMatchingAnnError(DatasetMergeError):
    item_id = field()
    ann = field()

    def __str__(self):
        return "Item %s: can't find matching annotation in sources %s, annotation is %s" % (
            self.item_id,
            self.sources,
            self.ann,
        )


@define(auto_exc=False)
class NoMatchingItemError(DatasetMergeError):
    item_id = field()

    def __str__(self):
        return "Item %s: can't find matching item in sources %s" % (self.item_id, self.sources)


@define(auto_exc=False)
class FailedLabelVotingError(DatasetMergeError):
    item_id = field()
    votes = field()
    ann = field(default=None)

    def __str__(self):
        return "Item %s: label voting failed%s, votes %s, sources %s" % (
            self.item_id,
            "for ann %s" % self.ann if self.ann else "",
            self.votes,
            self.sources,
        )


@define(auto_exc=False)
class FailedAttrVotingError(DatasetMergeError):
    item_id = field()
    attr = field()
    votes = field()
    ann = field()

    def __str__(self):
        return "Item %s: attribute voting failed for ann %s, votes %s, sources %s" % (
            self.item_id,
            self.ann,
            self.votes,
            self.sources,
        )


@define(auto_exc=False)
class VideoMergeError(DatasetMergeError):
    item_id = field()

    def __str__(self):
        return "Item %s: video merging is not possible" % (self.item_id,)


@define(auto_exc=False)
class DatasetValidationError(DatumaroError):
    severity = field()

    def to_dict(self):
        return {
            "anomaly_type": self.__class__.__name__,
            "description": str(self),
            "severity": self.severity.name,
        }


@define(auto_exc=False)
class DatasetItemValidationError(DatasetValidationError):
    item_id = field()
    subset = field()

    def to_dict(self):
        dict_repr = super().to_dict()
        dict_repr["item_id"] = self.item_id
        dict_repr["subset"] = self.subset
        return dict_repr


@define(auto_exc=False)
class MissingLabelCategories(DatasetValidationError):
    def __str__(self):
        return "Metadata (ex. LabelCategories) should be defined to validate a dataset."


@define(auto_exc=False)
class MissingAnnotation(DatasetItemValidationError):
    ann_type = field()

    def __str__(self):
        return f"Item needs '{self.ann_type}' annotation(s), but not found."


@define(auto_exc=False)
class BrokenAnnotation(DatasetItemValidationError):
    ann_type = field()

    def __str__(self):
        return f"Item needs whole '{self.ann_type}' annotation(s), but missed some."


@define(auto_exc=False)
class EmptyLabel(DatasetItemValidationError):
    label_name = field()

    def __str__(self):
        return f"Item should have the label '{self.label_name}' annotation(s), but not found."


@define(auto_exc=False)
class EmptyCaption(DatasetItemValidationError):
    caption_name = field()

    def __str__(self):
        return f"Item should have the caption '{self.caption_name}' annotation(s), but not found."


@define(auto_exc=False)
class MultiLabelAnnotations(DatasetItemValidationError):
    def __str__(self):
        return "Item needs a single label but multiple labels are found."


@define(auto_exc=False)
class MissingAttribute(DatasetItemValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return f"Item needs the attribute '{self.attr_name}' for the label '{self.label_name}'."


@define(auto_exc=False)
class UndefinedLabel(DatasetItemValidationError):
    label_name = field()

    def __str__(self):
        return f"Item has the label '{self.label_name}' which is not defined in metadata."


@define(auto_exc=False)
class UndefinedAttribute(DatasetItemValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return (
            f"Item has the attribute '{self.attr_name}' for the "
            f"label '{self.label_name}' which is not defined in metadata."
        )


@define(auto_exc=False)
class LabelDefinedButNotFound(DatasetValidationError):
    label_name = field()

    def __str__(self):
        return f"The label '{self.label_name}' is defined in metadata, but not found in the dataset."


@define(auto_exc=False)
class AttributeDefinedButNotFound(DatasetValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return (
            f"The attribute '{self.attr_name}' for the label "
            f"'{self.label_name}' is defined in metadata, but not "
            "found in the dataset."
        )


@define(auto_exc=False)
class OnlyOneLabel(DatasetValidationError):
    label_name = field()

    def __str__(self):
        return f"The dataset has only one label '{self.label_name}'."


@define(auto_exc=False)
class OnlyOneAttributeValue(DatasetValidationError):
    label_name = field()
    attr_name = field()
    value = field()

    def __str__(self):
        return (
            "The dataset has the only attribute value "
            f"'{self.value}' for the attribute '{self.attr_name}' for the "
            f"label '{self.label_name}'."
        )


@define(auto_exc=False)
class FewSamplesInLabel(DatasetValidationError):
    label_name = field()
    count = field()

    def __str__(self):
        return f"The number of samples in the label '{self.label_name}' might be too low. Found '{self.count}' samples."


@define(auto_exc=False)
class FewSamplesInCaption(DatasetValidationError):
    caption_name = field()
    count = field()

    def __str__(self):
        return (
            f"The number of samples in the caption '{self.caption_name}'"
            f" might be too low. Found '{self.count}' samples."
        )


@define(auto_exc=False)
class RedundanciesInCaption(DatasetValidationError):
    caption_name = field()
    redundancy_type = field()
    count = field()

    def __str__(self):
        return (
            f"The number of '{self.redundancy_type}' redundancy in the caption '{self.caption_name}'"
            f" have found '{self.count}'."
        )


@define(auto_exc=False)
class FewSamplesInAttribute(DatasetValidationError):
    label_name = field()
    attr_name = field()
    attr_value = field()
    count = field()

    def __str__(self):
        return (
            "The number of samples for attribute = value "
            f"'{self.attr_name} = {self.attr_value}' for the label "
            f"'{self.label_name}' might be too low. "
            f"Found '{self.count}' samples."
        )


@define(auto_exc=False)
class ImbalancedLabels(DatasetValidationError):
    def __str__(self):
        return "There is an imbalance in the label distribution."


@define(auto_exc=False)
class ImbalancedCaptions(DatasetValidationError):
    def __str__(self):
        return "There is an imbalance in the caption distribution."


@define(auto_exc=False)
class ImbalancedAttribute(DatasetValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return (
            "There is an imbalance in the distribution of attribute"
            f" '{self.attr_name}' for the label '{self.label_name}'."
        )


@define(auto_exc=False)
class ImbalancedDistInLabel(DatasetValidationError):
    label_name = field()
    prop = field()

    def __str__(self):
        return f"Values of '{self.prop}' are not evenly distributed for '{self.label_name}' label."


@define(auto_exc=False)
class ImbalancedDistInCaption(DatasetValidationError):
    caption_name = field()

    def __str__(self):
        return f"Values are not evenly distributed for '{self.caption_name}' caption."


@define(auto_exc=False)
class ImbalancedDistInAttribute(DatasetValidationError):
    label_name = field()
    attr_name = field()
    attr_value = field()
    prop = field()

    def __str__(self):
        return (
            f"Values of '{self.prop}' are not evenly "
            f"distributed for '{self.attr_name}' = '{self.attr_value}' for "
            f"the '{self.label_name}' label."
        )


@define(auto_exc=False)
class NegativeLength(DatasetItemValidationError):
    ann_id = field()
    prop = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.ann_id}' in "
            "the item should have a positive value of "
            f"'{self.prop}' but got '{self.val}'."
        )


@define(auto_exc=False)
class InvalidValue(DatasetItemValidationError):
    ann_id = field()
    prop = field()

    def __str__(self):
        return f"Annotation '{self.ann_id}' in the item has an inf or a NaN value of '{self.prop}'."


@define(auto_exc=False)
class FarFromLabelMean(DatasetItemValidationError):
    label_name = field()
    ann_id = field()
    prop = field()
    mean = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.ann_id}' in "
            f"the item has a value of '{self.prop}' that "
            "is too far from the label average. (mean of "
            f"'{self.label_name}' label: {self.mean}, got '{self.val}')."
        )


@define(auto_exc=False)
class FarFromCaptionMean(DatasetItemValidationError):
    caption_name = field()
    mean = field()
    upper_bound = field()
    lower_bound = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.caption_name}' in "
            "the item is too far from the caption average. (mean of "
            f"'{self.caption_name}' caption: '{self.mean}', got '{self.val}')."
            f"It should be between '{self.lower_bound}' and '{self.upper_bound}'."
        )


@define(auto_exc=False)
class OutlierInCaption(DatasetItemValidationError):
    caption_name = field()
    lower_bound = field()
    upper_bound = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.caption_name}' in "
            "the item is estimated as outlier based on IQR. (lower and upper bound of "
            f"'{self.caption_name}' caption: '{self.lower_bound}' and '{self.upper_bound}', got '{self.val}')."
        )


@define(auto_exc=False)
class FarFromAttrMean(DatasetItemValidationError):
    label_name = field()
    ann_id = field()
    attr_name = field()
    attr_value = field()
    prop = field()
    mean = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.ann_id}' in the "
            f"item has a value of '{self.prop}' that "
            "is too far from the attribute average. (mean of "
            f"'{self.attr_name}' = '{self.attr_value}' for the "
            f"'{self.label_name}' label: {self.mean}, got '{self.val}')."
        )


@define(auto_exc=False)
class StreamedItemError(DatasetError):
    def __str__(self):
        return "__getitem__ is disabled for StreamDataset. Use Dataset instead."
