<script>
	import SamplePlot from "../plots/SamplePlot.svelte";
	import { store } from "../store/store";
	import FileReceiver from "./FileReceiver.svelte";
	import PredictionsTable from "./PredictionsTable.svelte";
	import SummaryDetails from "./SummaryDetails.svelte";

  let viewMode;
  let predictions;

  const unsubscribe = store.subscribe((state) => {
    viewMode = state.viewMode;
    predictions = state.predictions;
  });

</script>

<div class="hero min-h-screen">
  {#if viewMode === 'selectMode'}
    <div class="hero-content text-center">
      <div class="max-w-md">
        <h1 class="text-5xl font-bold">Hello there</h1>
        <p class="py-6">Upload a csv file in smiles notation below</p>
        <FileReceiver />
      </div>
    </div>
  {:else if viewMode === 'summaryMode'}
    <div>
      <PredictionsTable predictions={predictions} /> 
      <div class="hero-content flex-col lg:flex-row">
        <SamplePlot />
        <SummaryDetails />
      </div>
    </div>
  {:else if viewMode === 'loadingMode'}
    <div class='predictions-placeholder-container'>
      <h3 class="text-3xl font-bold">Getting predictions...</h3>
      <progress class="progress w-56"></progress>
    </div>
  {/if}

</div>
