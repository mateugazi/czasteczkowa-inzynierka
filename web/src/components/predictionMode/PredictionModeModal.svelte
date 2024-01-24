<script>
	import { store } from "../../store/store";
	import { getPredictions } from "../api/getPredictions";

	let files;
	let models = [];
	let selectedModel = "";

	store.subscribe((state) => {
		models = state.models;
	});

	async function onTriggerCalculations() {
		store.update((state) => ({ ...state, viewMode: "loadingMode" }));
		const fetchedPredictions = await getPredictions(selectedModel, files[0]);
		store.update((state) => ({
			...state,
			viewMode: "summaryMode",
			predictions: fetchedPredictions,
		}));
	}
</script>

<dialog id="prediction-mode-modal" class="modal">
	<div class="modal-box w-11/12 max-w-5xl">
		<h2 class="font-bold text-3xl">Get predictions</h2>
		<div class="prediction-modal-body">
			<div class="models-section">
				<h2 class="font-bold text-2xl">Select model</h2>
				{#if models && models.length > 0}
					{#each models as model}
						<div class="card w-96 bg-base-100 shadow-xl">
							<div class="card-body">
								<h2 class="card-title">{model.name}</h2>
								<p>{model.description}</p>
								<div class="card-actions justify-end">
									<button
										class="btn btn-primary"
										on:click={() => (selectedModel = model._id)}
									>
										{#if selectedModel === model._id}
											SELECTED
										{:else}
											Select
										{/if}
									</button>
								</div>
							</div>
						</div>
					{/each}
				{:else}
					<p class="py-4">Sorry, no models available</p>
				{/if}
			</div>

			<div class="data-info-section">
				<label class="form-control w-full max-w-xs">
					<div class="label">
						<span class="text-l">Upload your dataset in the csv file</span>
					</div>
					<input
						accept=".csv"
						bind:files
						type="file"
						class="file-input file-input-bordered file-input-primary w-full max-w-xs"
					/>
				</label>
			</div>
		</div>
		<div class="modal-action">
			<form method="dialog">
				<button class="btn">Close</button>
			</form>
			<button class="btn btn-primary" on:click={onTriggerCalculations}
				>Run predictions</button
			>
		</div>
	</div>
</dialog>

<style>
	p {
		text-align: start;
	}

	figure {
		padding-top: 5rem;
		padding-bottom: 5rem;
	}

	.card {
		width: 100%;
	}

	.prediction-modal-body {
		display: flex;
		gap: 5%;
	}

	.models-section,
	.data-info-section {
		width: 50%;
	}

	.models-section {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}
</style>
