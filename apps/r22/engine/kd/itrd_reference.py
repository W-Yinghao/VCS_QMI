"""Independent ITRD loss adapter, with explicit upstream-compatible semantics.

Paper: Miles et al., Information Theoretic Representation Distillation, BMVC 2022.
Reference inspected: roymiles/ITRD@5d08c6b33e601228252e11cc1756d9f1af04a547,
distiller_zoo/IT.py and helper/loops.py. This file is independently written from
the mathematical operations; it is not a redistributed copy of upstream code.

This is the feature loss, not an assertion that the whole training recipe has
been reproduced. The caller freezes the teacher and adds supervised loss.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


REFERENCE_GIT_SHA = "5d08c6b33e601228252e11cc1756d9f1af04a547"
REFERENCE_URL = "https://github.com/roymiles/ITRD/blob/" + REFERENCE_GIT_SHA + "/distiller_zoo/IT.py"


class ITRDComponents(NamedTuple):
    correlation: Tensor
    mutual: Tensor


class ITRDFeatureLoss(nn.Module):
    """Return the author's weighted sum of correlation and Gram losses.

    Compatibility details that matter:
    - Correlation always uses a trainable student-to-teacher linear projection.
    - The Gram branch uses that projection only if feature dimensions differ.
    - Per-coordinate sample std uses correction=1; correlation divides by B.
    - Teacher features receive no projection in either branch.
    - Default strict mode preserves the official formula on nondegenerate data
      and fails explicitly on zero variance, rather than silently adding epsilon.

    std_floor>0 is an opt-in numerical variant and must be disclosed/tuned fairly.
    It does not change outputs whenever every sample std is above that floor.
    """

    def __init__(self, student_dim: int, teacher_dim: int, *, alpha: float = 1.01,
                 correlation_weight: float = 2.0, mutual_weight: float = 1.0,
                 std_floor: float = 0.0) -> None:
        super().__init__()
        for name, dim in (("student_dim", student_dim), ("teacher_dim", teacher_dim)):
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, val in (("alpha", alpha), ("correlation_weight", correlation_weight),
                          ("mutual_weight", mutual_weight), ("std_floor", std_floor)):
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if alpha == 0:
            raise ValueError("alpha must be positive")
        self.student_dim, self.teacher_dim = student_dim, teacher_dim
        self.alpha = float(alpha)
        self.correlation_weight, self.mutual_weight = float(correlation_weight), float(mutual_weight)
        self.std_floor = float(std_floor)
        self.projector = nn.Linear(student_dim, teacher_dim)

    def _validate(self, student: Tensor, teacher: Tensor) -> tuple[Tensor, Tensor]:
        if student.ndim < 2 or teacher.ndim < 2:
            raise ValueError("ITRD expects batch-first feature tensors")
        student, teacher = student.flatten(1), teacher.flatten(1)
        if len(student) != len(teacher) or len(student) < 2:
            raise ValueError("ITRD requires equal paired batches of at least two examples")
        if student.shape[1] != self.student_dim or teacher.shape[1] != self.teacher_dim:
            raise ValueError("Input dimensions do not match the configured feature taps")
        if student.device != teacher.device or student.dtype != teacher.dtype:
            raise ValueError("Student and teacher features must share dtype and device")
        if student.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use fp32/fp64 for the ITRD loss; disable AMP for this branch")
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise FloatingPointError("Nonfinite ITRD input features")
        return student, teacher

    def _standardize(self, values: Tensor) -> Tensor:
        scale = values.std(dim=0, correction=1)
        if self.std_floor:
            scale = scale.clamp_min(self.std_floor)
        elif bool((scale == 0).any()):
            raise FloatingPointError("Official ITRD formula is undefined at zero feature variance; record failure or explicitly enable the std_floor variant")
        return (values - values.mean(dim=0)) / scale

    def components(self, student: Tensor, teacher: Tensor) -> ITRDComponents:
        student, teacher = self._validate(student, teacher)
        projected = self.projector(student)
        sample_corr = (self._standardize(projected) * self._standardize(teacher)).mean(dim=0)
        correlation = torch.log2((sample_corr - 1).abs().pow(2 * self.alpha).sum())

        represented_student = projected if self.student_dim != self.teacher_dim else student
        unit_student = F.normalize(represented_student, dim=1, eps=1e-12)
        unit_teacher = F.normalize(teacher, dim=1, eps=1e-12)
        gram_student = unit_student @ unit_student.T
        gram_joint = gram_student * (unit_teacher @ unit_teacher.T)
        trace_student, trace_joint = gram_student.trace(), gram_joint.trace()
        if bool(trace_student <= 0) or bool(trace_joint <= 0):
            raise FloatingPointError("Official ITRD Gram trace normalization is undefined on this batch")
        mutual = (gram_student / trace_student).square().sum() - (gram_joint / trace_joint).square().sum()
        if not torch.isfinite(correlation) or not torch.isfinite(mutual):
            raise FloatingPointError("Nonfinite ITRD component")
        return ITRDComponents(correlation, mutual)

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        values = self.components(student, teacher)
        return self.correlation_weight * values.correlation + self.mutual_weight * values.mutual
