<script>
	export let parameterLabel = "Default label";
	export let placeholder = "Default placeholder";
	export let shouldPrefill = true;
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
		list={shouldPrefill && parameterLabel}
	/>
	<datalist id={parameterLabel}>
		<option value={placeholder} />
	</datalist>
</label>
