<script>
	import { fetchPredictions } from "../../helpers/fetchPredictions";
	import { store } from "../../store/store";

  let files;

  async function onTriggerCalculations() {
    store.update((state) => ({...state, viewMode: 'loadingMode'}));
    const fetchedPredictions = await fetchPredictions(files[0]);
    store.update((state) => ({...state, viewMode: 'summaryMode', predictions: fetchedPredictions.predictions}));
  }

</script>

<div class="file-receiver-container">
  <input accept=".csv" bind:files type="file" class="file-input file-input-bordered file-input-primary w-full max-w-xs" />
  {#if files}
    <button class="btn btn-primary btn-lg" on:click={onTriggerCalculations}>Run predictions</button>
  {/if}
</div>

<style>
  .btn {
    margin-top: 30px;
  }

  .file-receiver-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
</style>

