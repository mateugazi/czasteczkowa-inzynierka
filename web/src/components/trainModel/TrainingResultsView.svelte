<script>
	import { onMount } from "svelte";
	import TrainResults from "./TrainResults.svelte";
	import { scrollToTop } from "../../helpers/scrollToTop";

	onMount(() => {
		scrollToTop();
	});
</script>

<div class="training-results-view">
	<div role="alert" class="alert alert-success">
		<svg
			xmlns="http://www.w3.org/2000/svg"
			class="stroke-current shrink-0 h-6 w-6"
			fill="none"
			viewBox="0 0 24 24"
			><path
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
			/></svg
		>
		<span>Model successfuly saved in the database!</span>
	</div>
	<TrainResults />
</div>

<style>
	.training-results-view {
		margin-bottom: 50px;
	}
</style>
