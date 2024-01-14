<script>
	export let parameterLabel;
	export let placeholder;
</script>

<label class="form-control w-full max-w-xs">
	<div class="label">
		<span class="label-text">{parameterLabel}</span>
	</div>
	<input
		type="text"
		{placeholder}
		class="input input-bordered w-full max-w-xs"
		on:change
		list={parameterLabel}
	/>
	<datalist id={parameterLabel}>
		<option value={placeholder} />
	</datalist>
</label>
