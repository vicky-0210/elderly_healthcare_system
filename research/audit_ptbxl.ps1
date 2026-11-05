$ErrorActionPreference = 'Stop'
$researchRoot = $PSScriptRoot
$metadataRoot = Join-Path $researchRoot 'metadata'
$statementText = Get-Content -LiteralPath (Join-Path $metadataRoot 'scp_statements.csv') -Raw
$statements = ('code' + $statementText) | ConvertFrom-Csv
$diagnosticCodes = @{}
foreach ($statement in $statements) {
    if ($statement.diagnostic -eq '1.0') { $diagnosticCodes[$statement.code] = $true }
}
$records = @(Import-Csv -LiteralPath (Join-Path $metadataRoot 'ptbxl_database.csv'))
$manifest = foreach ($record in $records) {
    # SCP dictionaries contain only simple quoted code keys and numeric values.
    $codes = $record.scp_codes.Replace("'", '"') | ConvertFrom-Json -AsHashtable
    $diagnoses = @($codes.Keys | Where-Object { $diagnosticCodes.ContainsKey($_) })
    $hasLVH = $codes.ContainsKey('LVH')
    $hasNorm = $codes.ContainsKey('NORM')
    $diagNormal = $hasNorm -and $diagnoses.Count -eq 1
    $strictNormal = $hasNorm -and @($codes.Keys | Where-Object { $_ -notin @('NORM', 'SR') }).Count -eq 0
    $noiseFlag = @('baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems' | Where-Object { -not [string]::IsNullOrWhiteSpace($record.$_) }).Count -gt 0
    [pscustomobject]@{
        ecg_id = $record.ecg_id
        patient_id = $record.patient_id
        fold = [int]$record.strat_fold
        split = if ([int]$record.strat_fold -le 8) { 'train' } elseif ([int]$record.strat_fold -eq 9) { 'validation' } else { 'test' }
        lvh_present = $hasLVH
        lvh_likelihood = if ($hasLVH) { [double]$codes['LVH'] } else { $null }
        norm_present = $hasNorm
        norm_likelihood = if ($hasNorm) { [double]$codes['NORM'] } else { $null }
        norm_diagnostic_only = $diagNormal
        norm_strict = $strictNormal
        lvh_only_diagnostic = $hasLVH -and $diagnoses.Count -eq 1
        contradictory_lvh_norm = $hasLVH -and $hasNorm
        any_quality_metadata_flag = $noiseFlag
        age_raw = $record.age
        sex = $record.sex
    }
}
$cohorts = [ordered]@{
    all = @($manifest)
    lvh_any = @($manifest | Where-Object lvh_present)
    lvh_without_norm = @($manifest | Where-Object { $_.lvh_present -and -not $_.norm_present })
    lvh_100_without_norm = @($manifest | Where-Object { $_.lvh_present -and $_.lvh_likelihood -eq 100 -and -not $_.norm_present })
    lvh_only_diagnostic = @($manifest | Where-Object lvh_only_diagnostic)
    norm_diagnostic_only = @($manifest | Where-Object norm_diagnostic_only)
    norm_strict = @($manifest | Where-Object norm_strict)
    norm_strict_100 = @($manifest | Where-Object { $_.norm_strict -and $_.norm_likelihood -eq 100 })
    non_lvh = @($manifest | Where-Object { -not $_.lvh_present })
    initial_lvh_vs_strict_norm = @($manifest | Where-Object { ($_.lvh_present -and -not $_.norm_present) -or $_.norm_strict })
    one_vs_rest_without_contradiction = @($manifest | Where-Object { -not $_.contradictory_lvh_norm })
}
$summaries = foreach ($entry in $cohorts.GetEnumerator()) {
    $members = $entry.Value
    [pscustomobject]@{
        cohort = $entry.Key
        recordings = $members.Count
        patients = @($members.patient_id | Sort-Object -Unique).Count
        train = @($members | Where-Object split -eq 'train').Count
        validation = @($members | Where-Object split -eq 'validation').Count
        test = @($members | Where-Object split -eq 'test').Count
        quality_metadata_flagged = @($members | Where-Object any_quality_metadata_flag).Count
    }
}
$foldCounts = foreach ($fold in 1..10) {
    $members = @($manifest | Where-Object fold -eq $fold)
    [pscustomobject]@{
        fold = $fold
        recordings = $members.Count
        patients = @($members.patient_id | Sort-Object -Unique).Count
        lvh = @($members | Where-Object lvh_present).Count
        strict_norm = @($members | Where-Object norm_strict).Count
    }
}
$leaks = @($manifest | Group-Object patient_id | Where-Object { @($_.Group.fold | Sort-Object -Unique).Count -gt 1 })
$ageSummary = foreach ($entry in $cohorts.GetEnumerator()) {
    $members = $entry.Value
    [pscustomobject]@{
        cohort = $entry.Key
        age_65plus_including_deidentified_older = @($members | Where-Object { $_.age_raw -ne '' -and [double]$_.age_raw -ge 65 }).Count
        age_over_89_encoded = @($members | Where-Object { $_.age_raw -ne '' -and [double]$_.age_raw -gt 200 }).Count
    }
}
$audit = [ordered]@{
    dataset = 'PTB-XL 1.0.3'
    downloaded_for_review_date = '2026-11-05'
    waveform_inspection_performed = $false
    cohort_definitions = [ordered]@{
        lvh_any = 'LVH key present, regardless of confidence or comorbid codes.'
        norm_diagnostic_only = 'NORM is the only code marked diagnostic in scp_statements.csv; other form/rhythm codes allowed.'
        norm_strict = 'NORM present and every code is either NORM or SR; no signal-quality exclusion.'
        likelihood_note = 'Zero means unknown confidence, not absence. All confidence values are descriptive annotations, not calibrated probabilities.'
        quality_metadata_note = 'Any nonblank quality field is flagged descriptively, including nonspecific text; this is not a validated quality classifier.'
    }
    cross_fold_patient_leaks = $leaks.Count
    contradictory_lvh_norm_records = @($manifest | Where-Object contradictory_lvh_norm).Count
    lvh_confidence_distribution = @($cohorts.lvh_any | Group-Object lvh_likelihood | Sort-Object { [double]$_.Name } | Select-Object Name,Count)
    cohorts = @($summaries)
    folds = @($foldCounts)
    age = @($ageSummary)
    source_hashes = @(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $metadataRoot 'ptbxl_database.csv'),(Join-Path $metadataRoot 'scp_statements.csv') | Select-Object Path,Hash)
}
$manifest | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $researchRoot 'cohort_manifest.csv')
$summaries | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $researchRoot 'cohort_counts.csv')
$audit | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $researchRoot 'dataset_audit.json') -Encoding utf8
$audit | ConvertTo-Json -Depth 8
