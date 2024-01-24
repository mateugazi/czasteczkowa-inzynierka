<script>
	import { store } from "../../store/store";
	import { triggerTraining } from "../api/triggerTraining";
	import ParameterInputField from "./ParameterInputField.svelte";

	let files;
	let areDefaultParametersChecked = true;
	let parametersValues = {};
	let modelName = "";
	let modelDescription = "";

	let modelArchitectures = [];
	let defaultArchitecturePlaceholder = "Select architecture type";
	let selectedModelArchitecture = defaultArchitecturePlaceholder;

	store.subscribe((state) => {
		modelArchitectures = state.modelArchitectures;
	});

	const updateParametersValues = (event) => {
		const parameterName = event.target.labels[0].textContent.trim();
		parametersValues[parameterName] = {
			value: event.target.value,
			...modelArchitectures
				.find(
					(modelArchitecture) =>
						modelArchitecture.name === selectedModelArchitecture
				)
				.parameters.find((parameter) => parameter.name === parameterName),
		};
	};

	const onSubmit = async () => {
		console.log(files[0], parametersValues, selectedModelArchitecture);
		store.update((state) => ({ ...state, viewMode: "loadingMode" }));
		const trainingResults = await triggerTraining(
			modelArchitectures.find(
				(modelArchitecture) =>
					modelArchitecture.name === selectedModelArchitecture
			),
			modelName,
			modelDescription,
			files[0],
			parametersValues
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
				bind:value={selectedModelArchitecture}
				class="select select-bordered w-full max-w-xs text-l"
			>
				<option disabled selected>{defaultArchitecturePlaceholder}</option>
				{#each modelArchitectures as modelArchitecture}
					<option>{modelArchitecture.name}</option>
				{/each}
			</select>
			<ParameterInputField
				parameterLabel={"Model name"}
				placeholder={"Please provide a unique model name"}
				shouldPrefill={false}
				on:change={(event) => (modelName = event.target.value)}
			/>
			<ParameterInputField
				parameterLabel={"Model description"}
				placeholder={"Please provide a model description"}
				shouldPrefill={false}
				on:change={(event) => (modelDescription = event.target.value)}
			/>
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

			{#if selectedModelArchitecture !== defaultArchitecturePlaceholder}
				<label class="label cursor-pointer default-parameters-checkbox">
					<h4 class="text-l">Use default parameters</h4>
					<input
						type="checkbox"
						bind:checked={areDefaultParametersChecked}
						class="checkbox"
					/>
				</label>

				{#if !areDefaultParametersChecked}
					{#each modelArchitectures.find((modelArchitecture) => modelArchitecture.name === selectedModelArchitecture).parameters as parameter}
						<ParameterInputField
							parameterLabel={parameter.name}
							placeholder={parameter.example}
							on:change={updateParametersValues}
						/>
					{/each}
				{/if}
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
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 5px;
		margin-top: 15px;
	}
</style>
