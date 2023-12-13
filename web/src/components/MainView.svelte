<script>
	import SamplePlot from "../plots/SamplePlot.svelte";
	import { store } from "../store/store";
	import FileReceiver from "./selectMode/FileReceiver.svelte";
	import SelectModeView from "./selectMode/SelectModeView.svelte";
	import SummaryView from "./summaryMode/SummaryView.svelte";

  let viewMode;
  let predictions;

  const unsubscribe = store.subscribe((state) => {
    viewMode = state.viewMode;
    predictions = state.predictions;
  });

</script>

<div class="hero min-h-screen">
  {#if viewMode === 'selectMode'}
    <SelectModeView />
  {:else if viewMode === 'predictionMode'}
    <FileReceiver />
  {:else if viewMode === 'summaryMode'}
    <SummaryView />
  {:else if viewMode === 'loadingMode'}
    <div class='predictions-placeholder-container'>
      <h3 class="text-3xl font-bold">Getting predictions...</h3>
      <progress class="progress w-56"></progress>
    </div>
  {/if}

</div>
