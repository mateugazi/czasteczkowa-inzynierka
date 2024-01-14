<script>
	import { triggerTraining } from "../../helpers/triggerTraining";
	import { store } from "../../store/store";
	import ParameterInputField from "./ParameterInputField.svelte";

	export let modelTypes = [];
	console.log(modelTypes);

	let files;
	let defaultArchitecturePlaceholder = "Select architecture type";
	let selectedModelType = defaultArchitecturePlaceholder;
	let parametersValues = {};

	const updateParametersValues = (event) => {
		const parameterName = event.target.labels[0].textContent.trim();
		parametersValues[parameterName] = {
			value: event.target.value,
			...modelTypes
				.find((modelType) => modelType.name === selectedModelType)
				.parameters.find((parameter) => parameter.name === parameterName),
		};
	};

	const onSubmit = async () => {
		console.log(files[0], parametersValues, selectedModelType);
		store.update((state) => ({ ...state, viewMode: "loadingMode" }));
		const trainingResults = await triggerTraining(
			files[0],
			parametersValues,
			modelTypes.find((modelType) => modelType.name === selectedModelType)
		);
		console.log(trainingResults);
		store.update((previousState) => ({
			...previousState,
			trainingResults: trainingResults.data,
			viewMode: "trainMode",
		}));
	};
</script>

<dialog id="train-model-modal" class="modal">
	<div class="modal-box w-11/12 max-w-5xl">
		<h2 class="font-bold text-3xl">Train a new model</h2>
		<div class="modal-content">
			<select
				bind:value={selectedModelType}
				class="select select-bordered w-full max-w-xs"
			>
				<option disabled selected>{defaultArchitecturePlaceholder}</option>
				{#each modelTypes as modelType}
					<option>{modelType.name}</option>
				{/each}
			</select>
			<label class="form-control w-full max-w-xs">
				<div class="label">
					<span class="label-text">Upload your dataset in the csv file</span>
				</div>
				<input
					accept=".csv"
					bind:files
					type="file"
					class="file-input file-input-bordered file-input-primary w-full max-w-xs"
				/>
			</label>

			{#if selectedModelType !== defaultArchitecturePlaceholder}
				{#each modelTypes.find((modelType) => modelType.name === selectedModelType).parameters as parameter}
					<ParameterInputField
						parameterLabel={parameter.name}
						placeholder={parameter.example}
						on:change={updateParametersValues}
					/>
				{/each}
			{/if}
		</div>
		<div class="modal-action">
			<form method="dialog">
				<button class="btn">Close</button>
			</form>
			<button class="btn" on:click={onSubmit}>Start training</button>
		</div>
	</div>
</dialog>

<style>
	.modal-content {
		margin-top: 15px;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 15px;
	}
</style>
