<script>
	import { fetchPredictions } from "../helpers/fetchPredictions";
	import SamplePlot from "../plots/SamplePlot.svelte";
  import FileReceiver from "./FileReceiver.svelte";
  import PredictionsTable from './PredictionsTable.svelte';
	import SummaryDetails from "./SummaryDetails.svelte";
  let wereCalculationsTriggered = false;
  let shouldShowCalculations = false;
  let predictions = [];

  async function onTriggerCalculations(event) {
    wereCalculationsTriggered = true;
    const fetchedPredictions = await fetchPredictions(event.detail.file);
    predictions = fetchedPredictions.predictions;
    shouldShowCalculations = true;
  }

</script>
<div class="hero min-h-screen">
  {#if !wereCalculationsTriggered}
    <div class="hero-content text-center">
      <div class="max-w-md">
        <h1 class="text-5xl font-bold">Hello there</h1>
        <p class="py-6">Upload a csv file in smiles notation below</p>
        <FileReceiver on:triggercalculations={onTriggerCalculations} />
      </div>
    </div>
  {:else if shouldShowCalculations}
    <div>
      <PredictionsTable predictions={predictions} /> 
      <div class="hero-content flex-col lg:flex-row">
        <SamplePlot />
        <SummaryDetails />
      </div>
    </div>
  {:else}
    <div class='predictions-placeholder-container'>
      <h3 class="text-3xl font-bold">Getting predictions...</h3>
      <progress class="progress w-56"></progress>
    </div>
  {/if}

</div>

<style>
  .predictions-placeholder-container {
    display: flex;
    flex-direction: column;
    row-gap: 20px;
  }

  .summary-plot-container {
    display: flex;
  }
</style>