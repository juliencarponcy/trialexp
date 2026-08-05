import { useEffect, useState } from 'react'

import '@mantine/core/styles.css'
import { Center, MantineProvider } from '@mantine/core'
import { Button, Select, Flex, Slider, Text, Stack, Group } from '@mantine/core'
import { Notifications, notifications } from '@mantine/notifications';
import axios from 'axios'
import Plot from 'react-plotly.js'

function App() {
  const [cohortList, setCohortList] = useState(['August', 'April'])
  const [animalIDList, setAnimalIDList] = useState(['A', 'B', 'C'])
  const [sessionIDList, setSessionIDList] = useState(['1', '2', '3'])
  const [depthShift, setDepthShift] = useState(0)
  const [cohort, setCohort] = useState<string | null>(null)
  const [animalID, setAnimalID] = useState<string | null>(null)
  const [sessionID, setSessionID] = useState<string | null>(null)
  const [plotData, setPlotData] = useState([])
  interface CellMetricsData {
    mean: number[];
    pos_y_bin: number[];
    count: number[];
    [key: string]: any;
  }

  const [cellMetricsData, setCellMetricsData] = useState<CellMetricsData | null>(null)
  const [trackDate, setTrackDate] = useState('')
  const [binSize, setBinSize] = useState(0);
  const [trajectoryInfoExists, setTrajectoryInfoExists] = useState(false);

  useEffect(() => {
    // a async function to fetch data
    const fetchData = async () => {
      try {
        const response = await fetch('/api/cohorts')
        if (!response.ok) {
          throw new Error('Server error');
        }
        const result = await response.json();
        setCohortList(result['cohorts'])
      } catch (err) {
        console.log('Error:', err);
      }
    }

    fetchData();
  }, []);

  useEffect(() => {
    // a async function to fetch data
    const fetchData = async () => {
      try {
        const response = await fetch(`/api/animal_id?cohort=${cohort}`)
        if (!response.ok) {
          throw new Error('Server error');
        }
        const result = await response.json();
        setAnimalID(null);
        setAnimalIDList(result['animal_id']);
      } catch (err) {
        console.log('Error:', err);
      }
    }

    fetchData();
  }, [cohort]);


  useEffect(() => {
    axios.get('/api/sessions',
      { params: { cohort: cohort, animal_id: animalID } })
      .then((response) => {
        setSessionIDList(response.data['session_id']);
        setSessionID(null);
      })
      .catch((error) => {
        console.log(error);
      })
      .finally(() => { });

  }, [cohort, animalID]);

  const fetchPlotData = () => {
    if (sessionID) {
      axios.get(`/api/trajectory/${sessionID}`,
        { params: { shift: depthShift } })
        .then((response) => {
          setPlotData(response.data);
          setTrackDate(response.data[0]['track_date'] ? response.data[0]['track_date'] : '');
        })
        .catch((error) => {
          console.log(error);
        });
    }
  };



  useEffect(() => {
    fetchPlotData();
  }, [sessionID, depthShift]);

  //Get firing rate data
  useEffect(() => {
    if (sessionID) {
      setDepthShift(0);
      axios.get(`/api/cell_metrics/${sessionID}`,
        { params: { bin_size: binSize } })
        .then((response) => {
          setCellMetricsData(response.data);
          if ('shift' in response.data) {
            setDepthShift(response.data['shift']);
            setTrajectoryInfoExists(true);
            notifications.show({
              title: 'Info',
              message: 'Existing shift data loaded',
            });
          } else {
            setTrajectoryInfoExists(false);
          }
        })
        .catch((error) => {
          console.log(error);
          const msg = error.response?.data?.detail ?? 'Failed to load cell metrics';
          notifications.show({
            title: 'Error',
            message: msg,
            color: 'red',
          });
          if (error.response?.status === 404) {
            setCellMetricsData(null);
            setPlotData([]);
            setTrackDate('');
          }
        });
    }
  }, [sessionID, binSize])

  const handleSave = () => {
    axios.post('/api/save_shift', { shift: depthShift, session_id: sessionID })
      .then(() => {
        notifications.show({
          title: 'Info',
          message: 'Shift info saved successfully',
        });

        setTrajectoryInfoExists(true);

      })
      .catch((error) => {
        console.log('Error saving depth shift:', error);
      });
  };

  const plotTraces = plotData.map((region: any) => ({
    x: ['Brain Regions'],
    y: [region.depth_start - region.depth_end],
    base: [region.depth_end],
    name: region.acronym,
    type: 'bar',
    text: region.name,
    textposition: 'inside'
  }));

  const frPlotTraces = cellMetricsData ? [{
    x: cellMetricsData['mean'],
    y: cellMetricsData['pos_y_bin'],
    type: 'bar',
    orientation: 'h'
  }] : [];

  const cellCountTraces = cellMetricsData && Object.keys(cellMetricsData).length > 0 ? [{
    x: cellMetricsData['count'],
    y: cellMetricsData['pos_y_bin'],
    type: 'bar',
    orientation: 'h'
  }] : [];

  return (
    <>
      <MantineProvider>
      <Notifications />
        <Stack>
          <Flex justify="center" align="flex-end" gap="md">
            <Select data={cohortList} placeholder="Cohort"
              label='Cohort'
              value={cohort} onChange={setCohort} searchable />
            <Select data={animalIDList} placeholder="Animal" label='Animal'
              value={animalID} onChange={setAnimalID} searchable />
            <Select data={sessionIDList} placeholder="Session" label='Session'
              value={sessionID} onChange={setSessionID} searchable />
          </Flex>

          <Center>
            <Group gap="xl">
              <Stack>
                <Text> Probe depth shift</Text>
                <Group>
                  <Slider
                    value={depthShift}
                    onChange={setDepthShift}
                    min={-2000}
                    max={2000}
                    labelAlwaysOn
                    style={{ width: 600 }} />
                  <Button onClick={() => setDepthShift(0)} disabled={cellMetricsData === null}>Reset</Button>
                  <Button
                    onClick={handleSave}
                    color={trajectoryInfoExists ? 'red' : 'blue'}
                    disabled={cellMetricsData === null}
                  >
                    {trajectoryInfoExists ? 'Overwrite' : 'Save'}
                  </Button>
                </Group>
              </Stack>

              <Stack>
                <Text>Bin size: {binSize}um</Text>
                <Group>
                  <Slider
                    value={binSize}
                    onChange={setBinSize}
                    min={0}
                    max={200}
                    step={20}
                    style={{ width: 200 }}
                  />
                </Group>
              </Stack>
            </Group>

          </Center>


        </Stack>

        <Center>
          <Group>
            <Plot
              data={plotTraces}
              layout={{
                title: `Mapped trajectory<br> ${trackDate}`,
                barmode: 'stack',
                xaxis: { title: 'Brain Regions' },
                yaxis: { title: 'Distance from tip (µm)', range: [0, 4000] },
                legend: { traceorder: 'normal' },
                width: 400,
                height: 1000,
                base: 1000,
              }}
            />

            <Plot
              data={frPlotTraces}
              layout={{
                title: 'Firing Rate by Position (experiment)',
                xaxis: { title: 'Firing Rate (Hz)' },
                yaxis: { title: 'Distance from tip (µm)', range: [0, 4000] },
                width: 400,
                height: 1000,
                orientation: 'h'
              }}
            />
            <Plot
              data={cellCountTraces}
              layout={{
                title: 'Cell count',
                xaxis: { title: 'Cell count' },
                yaxis: { title: 'Distance from tip (µm)', range: [0, 4000] },
                width: 400,
                height: 1000,
                orientation: 'h'
              }}
            />
          </Group>
        </Center>



      </MantineProvider>
    </>
  )
}

export default App
