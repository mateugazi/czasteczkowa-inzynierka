<script>
  import { store } from "../../store/store";
import FileReceiver from "./FileReceiver.svelte";
	import ModeCard from "./ModeCard.svelte";
	import { predictionIconPath, predictionsDescription, predictionsTitle, retrainModelDescription, retrainModelIconPath, retrainModelTitle, trainNewModelDescription, trainNewModelIconPath, trainNewModelTitle } from "./data/descriptions";
</script>

<div class="hero min-h-screen">
  <div class="hero-content text-center">
    <div class="max-w-lg">
      <h1 class="text-5xl font-bold">Welcome back</h1>
      <p class="py-6">Please choose one of the following workflows</p>
      <div class='select-workflow-container'>
        <ModeCard title={retrainModelTitle} description={retrainModelDescription} iconPath={retrainModelIconPath} />
        <ModeCard title={trainNewModelTitle} description={trainNewModelDescription} iconPath={trainNewModelIconPath} />
        <ModeCard 
          title={predictionsTitle} 
          description={predictionsDescription} 
          iconPath={predictionIconPath} 
          onSelect={() => store.update((previousState) => ({...previousState, viewMode: 'predictionMode'}))} 
        />
      </div>
    </div>
  </div>
</div>

<style>
  .select-workflow-container {
    display: flex;
    gap: 1rem;
  }

  .max-w-lg {
    max-width: 70rem;
  }
</style>