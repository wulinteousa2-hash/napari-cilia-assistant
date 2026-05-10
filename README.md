# napari-cilia-assistant

`napari-cilia-assistant` is a napari plugin for exploratory ciliary motion analysis from high-speed AVI microscopy videos.

The plugin supports ROI-based ciliary beat frequency (CBF) measurement, kymograph review, spatial CBF heatmaps, motion activity maps, and experimental optical-flow descriptors.

[Watch the demo video](docs/ui.mp4)

## Install

Install conda-forge first:  
https://conda-forge.org/download/

Then open a terminal and run:

```bash
conda create -n cilia-assistant python=3.11 -y
conda activate cilia-assistant
conda install -c conda-forge napari pyqt git -y
git clone https://github.com/wulinteousa2-hash/napari-cilia-assistant.git
cd napari-cilia-assistant
pip install -e .
napari
```

In napari, open:

```text
Plugins > Cilia Assistant
```

## What It Does

- Opens one AVI video through the napari widget.
- Reads video metadata such as FPS, frame count, size, codec, and duration.
- Loads AVI data as a grayscale `T, Y, X` image stack.
- Lets the user draw/edit a rectangular ROI over active cilia.
- Supports optional background ROI subtraction.
- Measures ROI mean-intensity change over time.
- Estimates CBF using FFT, Welch PSD, or periodogram analysis.
- Provides a peak-interval CBF check.
- Creates a kymograph layer from the selected ROI.
- Generates spatial CBF heatmaps.
- Generates motion activity maps to help locate moving regions.
- Provides experimental optical-flow maps for apparent motion direction, magnitude, curl, and deformation.
- Exports the last ROI signal and frequency spectrum as CSV files.
- Copies or saves the current measurement graph.

## User Interface Overview

The widget uses a five-step workflow:

1. **Input**  
   Load an AVI file, inspect metadata, and confirm FPS.

2. **Region of Interest**  
   Draw or edit the cilia ROI. Optionally define a background ROI.

3. **Analysis**  
   Choose one of the analysis tabs:

   - **ROI Frequency**  
     Standard CBF measurement from a selected ROI. This is the main workflow for quantitative reporting.

   - **CBF Heatmap**  
     Generates a spatial map of dominant frequency across the selected ROI or whole frame.

   - **Motion Activity**  
     Shows where the video changes over time. This is useful for finding active cilia, drift, debris, or non-ciliary motion.

   - **Advanced Flow**  
     Experimental optical-flow analysis for apparent motion magnitude, direction, curl, and deformation.

4. **Results / Graphs**  
   Review the intensity trace, frequency spectrum, peak result, heatmap, activity map, or flow-map summary.

5. **Export & Log**  
   Export results and copy the analysis log.

## Basic ROI Frequency Workflow

1. Open napari.
2. Open `Plugins > Cilia Assistant`.
3. Click **Open AVI**.
4. Confirm the FPS. Correct it manually if the AVI metadata are wrong.
5. Click **Create / Edit ROI Rectangle**.
6. Move/resize the ROI over visibly beating cilia.
7. Optional: create a background ROI if there is shared illumination or focus drift.
8. Go to **Step 3 > ROI Frequency**.
9. Choose the frequency method:
   - `FFT` for simple dominant-frequency analysis.
   - `Welch` for noisier traces.
   - `Periodogram` as another spectrum-based check.
10. Set the expected CBF search range, for example `3-25 Hz`.
11. Click **Analyze Selected ROI**.
12. Review the trace, frequency peak, peak-interval result, and kymograph.
13. Export CSV files if the result is usable.

## Output

- **Frequency CBF:** dominant frequency in the selected search range.
- **Peak-interval CBF:** independent check based on repeated peaks in the ROI signal.
- **Kymograph:** visual audit of periodic motion in the ROI.
- **CBF heatmap:** spatial map of estimated dominant frequency.
- **Peak-strength map:** map showing relative frequency-peak strength.
- **Motion activity map:** temporal motion/activity map.
- **Optical-flow maps:** exploratory apparent motion descriptors.
- **CSV export:** raw ROI time-intensity signal and frequency spectrum.
- **Graph export:** copy or save the current measurement graph.

## Good Measurement Practice

- Use videos with known FPS.
- Correct the FPS manually if AVI metadata are wrong.
- Keep temperature, medium, and timing consistent across samples.
- Place the ROI on active cilia, not static tissue, debris, or whole-frame motion.
- Use motion activity maps to identify candidate moving regions before final ROI measurement.
- Use CBF heatmaps as exploratory spatial screening, not as a replacement for careful ROI review.
- Use multiple ROIs/videos and biological replicates for group comparisons.
- Treat whole-frame frequency and optical-flow results as exploratory only.
- Always review the raw video, ROI placement, graph, and kymograph before interpreting the number.

## Limitations

This plugin measures ciliary motion from intensity changes in AVI microscopy videos. The standard ROI workflow estimates beat frequency, not full ciliary waveform or clinical diagnostic beat pattern.

A sample can have a normal CBF but abnormal waveform or poor flow generation. CBF heatmaps, motion activity maps, and optical-flow maps are useful exploratory tools, but they do not replace expert review of the raw video.

The **Advanced Flow** tab is experimental. Flow magnitude, direction, curl, and deformation should be interpreted as image-motion descriptors, not diagnostic classifications.


## Acknowledgements

During development, I reviewed publicly available open-source cilia motion-analysis resources, including the `cilia-metrics` repository:

https://github.com/quinngroup/cilia-metrics

The repository was useful for understanding existing computational approaches to ciliary motion analysis, especially spatial CBF mapping and frequency-domain analysis. This helped guide the addition of CBF heatmaps, Welch/PSD options, and motion-map style outputs in `napari-cilia-assistant`.

`napari-cilia-assistant` is independently implemented as a napari-based interactive workflow for AVI loading, ROI-based CBF measurement, kymograph review, spatial screening, and exportable analysis logs.

## References

1. Chilvers MA, O'Callaghan C. Analysis of ciliary beat pattern and beat frequency using digital high speed imaging: comparison with the photomultiplier and photodiode methods. *Thorax*. 2000;55:314-317. doi:10.1136/thorax.55.4.314

2. Jackson CL, Bottier M. Methods for the assessment of human airway ciliary function. *European Respiratory Journal*. 2022;60:2102300. doi:10.1183/13993003.02300-2021

3. Francis R. A Simple Method for Imaging and Quantifying Respiratory Cilia Motility in Mouse Models. *Methods and Protocols*. 2025;8:113. doi:10.3390/mps8050113
