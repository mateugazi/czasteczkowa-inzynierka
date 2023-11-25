<script>
	import { createEventDispatcher } from "svelte";
  let files;

	const dispatch = createEventDispatcher();
</script>

<input accept=".csv" bind:files type="file" class="file-input file-input-bordered file-input-primary w-full max-w-xs" />

{#if files}
  <button class="btn btn-primary btn-lg" on:click={() => dispatch('triggercalculations', {file: files[0]})}>Run predictions</button>
{/if}

<style>
  .btn {
    margin-top: 30px;
  }
</style>

