<script>
	import { fetchPredictions } from "../helpers/fetchPredictions";
	import { store } from "../store/store";
  let files;

  async function onTriggerCalculations() {
    store.update((state) => ({...state, viewMode: 'loadingMode'}));
    const fetchedPredictions = await fetchPredictions(files[0]);
    store.update((state) => ({...state, viewMode: 'summaryMode', predictions: fetchedPredictions.predictions}));
  }

</script>

<input accept=".csv" bind:files type="file" class="file-input file-input-bordered file-input-primary w-full max-w-xs" />

{#if files}
  <button class="btn btn-primary btn-lg" on:click={onTriggerCalculations}>Run predictions</button>
{/if}

<style>
  .btn {
    margin-top: 30px;
  }
</style>

