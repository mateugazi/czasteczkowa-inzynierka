<script>
	import { onMount } from "svelte";
	import { store } from "../../store/store";
	import { scrollToTop } from "../../helpers/scrollToTop";
	import { getPredictions } from "../api/getPredictions";

	let files;

	onMount(() => {
		scrollToTop();
	});

	async function onTriggerCalculations() {
		store.update((state) => ({ ...state, viewMode: "loadingMode" }));
		const fetchedPredictions = await getPredictions(files[0]);
		store.update((state) => ({
			...state,
			viewMode: "summaryMode",
			predictions: fetchedPredictions,
		}));
	}
</script>

<div class="file-receiver-container">
	<p class="py-6 text-3xl">Upload a csv file in smiles notation below</p>
	<input
		accept=".csv"
		bind:files
		type="file"
		class="file-input file-input-bordered file-input-primary w-full max-w-xs"
	/>
	{#if files}
		<button class="btn btn-primary btn-lg" on:click={onTriggerCalculations}
			>Run predictions</button
		>
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
		align-items: center;
	}
</style>
