<script>
	import { store } from "../store/store";
	import FileReceiver from "./selectMode/FileReceiver.svelte";
	import SelectModeView from "./selectMode/SelectModeView.svelte";
	import SummaryView from "./summaryMode/SummaryView.svelte";
	import TraininingResultsView from "./trainModel/TrainingResultsView.svelte";

	let viewMode;

	const unsubscribe = store.subscribe((state) => {
		viewMode = state.viewMode;
	});
</script>

<div class="hero min-h-screen">
	{#if viewMode === "selectMode"}
		<SelectModeView />
	{:else if viewMode === "predictionMode"}
		<FileReceiver />
	{:else if viewMode === "summaryMode"}
		<SummaryView />
	{:else if viewMode === "loadingMode"}
		<div class="predictions-placeholder-container">
			<h3 class="text-3xl font-bold">Loading...</h3>
			<progress class="progress w-56"></progress>
		</div>
	{:else if viewMode === "trainMode"}
		<TraininingResultsView />
	{/if}
</div>
