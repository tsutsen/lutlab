lutlab.segmented
================

Segmented (promptable) LUTs -- the Art 5 pipeline
``Y(x) = sum_s m_s(x) P(x; theta_s)``. The one thing that varies across papers
is how the region masks ``m_s`` are made, so segmentation is a pluggable
``backend`` (K-means by default; frozen SAM is a stub you register in).

.. autosummary::

   lutlab.segmented.segment_image
   lutlab.segmented.register_segmenter
   lutlab.segmented.available_backends
   lutlab.segmented.fit_with_masks
   lutlab.segmented.fit_and_apply

**Re-exported from ``robust_hald``**

.. autosummary::

   lutlab.segmented.fit_segmented
   lutlab.segmented.segmented_forward
   lutlab.segmented.SegmentedLUT
   lutlab.segmented.bake_lut
   lutlab.segmented.bake_segmented_luts
   lutlab.segmented.segmented_apply

.. automodule:: lutlab.segmented
   :members:
   :imported-members:
