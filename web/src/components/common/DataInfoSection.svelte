<script>
	export let files;
</script>

<div class="data-info-section">
	<h2 class="font-bold text-2xl">Upload your dataset in the csv file</h2>
	<label class="form-control w-full max-w-xs">
		<div class="label"></div>
		<input
			accept=".csv"
			bind:files
			type="file"
			class="file-input file-input-bordered file-input-primary w-full max-w-xs"
		/>
	</label>
	<div role="alert" class="alert alert-info">
		<svg
			xmlns="http://www.w3.org/2000/svg"
			fill="none"
			viewBox="0 0 24 24"
			class="stroke-current shrink-0 w-6 h-6"
			><path
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
			></path></svg
		>
		<span
			>Please make sure to have SMILES in <span class="message-italic">mol</span
			> column. Otherwise the workflow will not work.</span
		>
	</div>
</div>

<style>
	.data-info-section {
		width: 50%;
		display: flex;
		flex-direction: column;
		gap: 20px;
		align-items: center;
	}

	.message-italic {
		font-style: italic;
	}

	.alert-info {
		width: 75%;
	}
</style>
