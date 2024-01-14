<script>
	import { store } from "../../store/store";
	import TrainResults from "./TrainResults.svelte";

	let trainingResults;

	const unsubscribe = store.subscribe((state) => {
		trainingResults = state.trainingResults;
	});
</script>

<div class="training-results-view">
	<TrainResults {trainingResults} />
</div>

<style>
	.training-results-view {
		max-width: 80vw;
	}
</style>
