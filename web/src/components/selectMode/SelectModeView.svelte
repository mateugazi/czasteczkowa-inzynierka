<script>
	import PredictionModeCard from "../predictionMode/PredictionModeCard.svelte";
	import TrainModelCard from "../trainModel/TrainModelCard.svelte";
	import ModeCard from "./ModeCard.svelte";
	import {
		predictionIconPath,
		predictionsDescription,
		predictionsTitle,
		retrainModelDescription,
		retrainModelIconPath,
		retrainModelTitle,
		trainNewModelDescription,
		trainNewModelIconPath,
		trainNewModelTitle,
	} from "./data/descriptions";
</script>

<div class="hero min-h-screen">
	<div class="hero-content text-center">
		<div class="max-w-lg">
			<h1 class="text-5xl font-bold">Welcome back</h1>
			<p class="text-2xl py-6">Please choose one of the following workflows</p>
			<div class="py-6 select-workflow-container">
				<ModeCard
					title={retrainModelTitle}
					description={retrainModelDescription}
					iconPath={retrainModelIconPath}
					viewMode={"predictionMode"}
				/>
				<TrainModelCard
					title={trainNewModelTitle}
					description={trainNewModelDescription}
					iconPath={trainNewModelIconPath}
					viewMode={"trainMode"}
				/>
				<PredictionModeCard
					title={predictionsTitle}
					description={predictionsDescription}
					iconPath={predictionIconPath}
					viewMode={"predictionMode"}
				/>
			</div>
		</div>
	</div>
</div>

<style>
	.select-workflow-container {
		display: flex;
		gap: 3rem;
	}

	.max-w-lg {
		max-width: 70rem;
	}
</style>
