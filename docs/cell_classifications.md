The classification of high-quality striatal single units into Medium Spiny Neurons (MSNs), Fast-Spiking Interneurons (FSIs), Tonically Active Neurons (TANs), and unidentified interneurons is based on sequential application of three primary electrophysiological factors.

Here is a table describing these factors and their meaning in the context of cell type classification:

| Factor | Description/Meaning | Application and Classification Criteria |
| :--- | :--- | :--- |
| **1. Waveform Duration** (Trough-to-Peak Duration) | The time taken for the Kilosort template waveform to go from its trough (minimum amplitude) to its peak (maximum amplitude). | This is the initial sorting step. **Narrow-waveform units** (typically **$\leq 400\ \mu\text{s}$**) are identified as either **FSIs or unidentified interneurons**. MSNs and TANs are the remaining units (presumably having wider waveforms). |
| **2. Proportion of Time Associated with Long Interspike-Intervals ($> 2\ \text{s}$)** | This ratio is calculated by summing all inter-spike intervals longer than $2\ \text{s}$ and dividing that sum by the total recording time. | This factor is used to separate the narrow-waveform units. Neurons for which this ratio was **more than 10%** are classified as **unidentified interneurons**. The remaining narrow-waveform units are classified as putative **FSIs**. |
| **3. Post-spike Suppression** | The length of time that a unit's firing rate remains suppressed following an action potential. | This factor is measured using the unit's **autocorrelation function** (correlogram). The suppression time is counted as the number of **1-ms bins** in the autocorrelation function until the firing rate is equal to or greater than its average firing rate over the 600-ms to 900-ms autocorrelation bins. |
| | | This factor separates the remaining units (MSNs and TANs). Units with post-spike suppression of **$> 40\ \text{ms}$ are labelled TANs**. The units that remain (those with short suppression) are labelled **MSNs**. |

**Note:** A small number of units (36 out of 8,303) exhibited both short waveforms ($<400\ \mu\text{s}$) and long post-spike suppression ($>40\ \text{ms}$). These rare units exhibited TAN-like responses but were excluded from further analysis.

