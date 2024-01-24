<script>
	import { onMount } from "svelte";
	import SamplePlot from "../../plots/SamplePlot.svelte";
	import { store } from "../../store/store";
	import PredictionsTable from "../PredictionsTable.svelte";
	import SummaryDetails from "../SummaryDetails.svelte";
	import { scrollToTop } from "../../helpers/scrollToTop";

	let predictions;

	onMount(() => {
		scrollToTop();
	});

	const unsubscribe = store.subscribe((state) => {
		predictions = state.predictions;
	});
</script>

<div class="summary-mode">
	<h1 class="text-5xl font-bold">Your results</h1>
	<PredictionsTable {predictions} />
	<div class="hero-content flex-col lg:flex-row">
		<SamplePlot />
		<SummaryDetails />
	</div>
</div>

<style>
	.summary-mode {
		margin-top: 20px;
		max-width: 90vw;
	}

	h1 {
		margin-bottom: 20px;
	}
</style>
