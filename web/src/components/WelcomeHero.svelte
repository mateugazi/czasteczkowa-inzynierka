<script>
	import FileReceiver from "./FileReceiver.svelte";
  let wereCalculationsTriggered = false;

  function onTriggerCalculations() {
    wereCalculationsTriggered = true;
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
</style>