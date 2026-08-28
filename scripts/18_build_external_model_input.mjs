import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

// This adapter maps the audited 2026 extraction to the locked 25-variable
// input contract. It never fills values that were not reported; missing SA
// remains null and is flagged through sa_missing for fold-wise imputation.
const [inputPath, outputPath, previewPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error(
    "Usage: node scripts/18_build_external_model_input.mjs INPUT_CORE.xlsx OUTPUT_MODEL_READY.xlsx [PREVIEW.png]"
  );
}
const rawFeatures = [
  "ph", "temp", "ce", "rpm", "sa", "fg_complexing", "fg_polar", "fg_any",
  "ags_aged", "ags_virgin", "ph_missing", "temp_missing", "rpm_missing", "sa_missing",
  "metal_cd", "metal_cr", "metal_hg", "ret_other", "ret_pa", "ret_pe", "ret_pet",
  "ret_pla", "ret_pp", "ret_ps", "ret_pvc",
];

const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const source = wb.worksheets.getItem("Raw_Extracted");
const sourceValues = source.getUsedRange().values;
const headers = sourceValues[0];
const index = Object.fromEntries(headers.map((h, i) => [h, i]));
const value = (row, name) => row[index[name]];
const data = sourceValues.slice(1).filter((row) => row[index.record_id] != null);

for (const name of ["Model_Input_25Features", "Mapping_Audit_25Features"]) {
  const existing = wb.worksheets.getItemOrNullObject(name);
  if (!existing.isNullObject) existing.delete();
}
const model = wb.worksheets.add("Model_Input_25Features");
const audit = wb.worksheets.add("Mapping_Audit_25Features");

const outputHeaders = ["record_id", "aut_id", "exp_id", "qe", ...rawFeatures, "mapping_status", "mapping_note"];
const modelRows = data.map((row) => {
  const study = String(value(row, "study_id"));
  const polymer = String(value(row, "polymer"));
  const aging = String(value(row, "aging_state"));
  const isPet = study === "EXT2026_PET_CD_01";
  const peUnaged = study === "EXT2026_PE_CD_02" && aging === "Unaged";
  const aged = !peUnaged;
  const ph = value(row, "pH");
  const temp = value(row, "temperature_C");
  const rpm = value(row, "rpm");
  const experiment = isPet ? "EXT2026_PET_UVC_336H" : `EXT2026_PE_${aging.replaceAll(" ", "_").toUpperCase()}`;
  const features = {
    ph, temp, ce: value(row, "Ce_mg_L"), rpm, sa: null,
    fg_complexing: aged ? 1 : 0,
    fg_polar: aged ? 1 : 0,
    fg_any: aged ? 1 : 0,
    ags_aged: aged ? 1 : 0,
    ags_virgin: aged ? 0 : 1,
    ph_missing: ph == null ? 1 : 0,
    temp_missing: temp == null ? 1 : 0,
    rpm_missing: rpm == null ? 1 : 0,
    sa_missing: 1,
    metal_cd: 1, metal_cr: 0, metal_hg: 0,
    ret_other: 0, ret_pa: 0, ret_pe: polymer === "PE" ? 1 : 0,
    ret_pet: polymer === "PET" ? 1 : 0, ret_pla: 0, ret_pp: 0, ret_ps: 0, ret_pvc: 0,
  };
  return [
    value(row, "record_id"), study, experiment, value(row, "qe_mg_g"),
    ...rawFeatures.map((field) => features[field]),
    "ready_with_documented_missingness",
    isPet
      ? "PET UV-C: pH/temp/rpm/Ce reported; SA not reported; aged oxygen-containing groups coded from article."
      : peUnaged
        ? "PE unaged: pH and SA not reported; no new oxygen-functional groups reported."
        : "PE naturally aged: pH and SA not reported; oxygen-containing groups/aging coded from article FTIR and methods.",
  ];
});

model.getRangeByIndexes(0, 0, modelRows.length + 1, outputHeaders.length).values = [outputHeaders, ...modelRows];
model.tables.add(`A1:AE${modelRows.length + 1}`, true, "ExternalModelInput25FeaturesTable");
model.showGridLines = false;
model.freezePanes.freezeRows(1);
model.getRange("A1:AE1").format = {fill: "#17365D", font: {bold: true, color: "#FFFFFF"}, wrapText: true, horizontalAlignment: "center"};
model.getRange(`D2:D${modelRows.length + 1}`).format.numberFormat = "0.000";
model.getRange(`E2:H${modelRows.length + 1}`).format.numberFormat = "0.000";
model.getRange("AD:AE").format.wrapText = true;
model.getRange("AE:AE").format.columnWidth = 80;
model.getRange("A:AC").format.autofitColumns();

const auditRows = [
  ["ph", "Raw_Extracted.pH", "7.5 reported", "Not reported; null", "Reported for PET only; ph_missing is explicit."],
  ["temp", "Raw_Extracted.temperature_C", "25 C", "25 C", "Directly reported."],
  ["ce", "Raw_Extracted.Ce_mg_L", "Digitized from Figure 8a", "Mass-balance-derived from C0 and digitized qe", "Directly usable equilibrium concentration."],
  ["rpm", "Raw_Extracted.rpm", "150", "130", "Directly reported."],
  ["sa", "No defensible measured value", "null", "null", "Do not substitute particle size or estimate BET area; sa_missing=1."],
  ["fg_complexing / fg_polar / fg_any", "Article FTIR and oxidation findings", "1 / 1 / 1", "Aged: 1 / 1 / 1; unaged: 0 / 0 / 0", "Aged PET: oxygen-containing groups; aged PE: carbonyl/hydroxyl groups. Binary coding is article-level, not a measured intensity."],
  ["ags_aged / ags_virgin", "Article methods", "1 / 0", "Aged: 1 / 0; unaged: 0 / 1", "Direct experimental condition."],
  ["missingness indicators", "Source completeness", "ph/temp/rpm 0; SA 1", "pH/SA 1; temp/rpm 0", "Matches locked-model missingness convention."],
  ["metal one-hot", "Cd(II) in both papers", "metal_cd=1", "metal_cd=1", "All other metal indicators 0."],
  ["polymer one-hot", "Article material", "ret_pet=1", "ret_pe=1", "All other polymer indicators 0."],
  ["not model features", "C0, contact time, mass, volume, particle size", "Retained in Raw_Extracted", "Retained in Raw_Extracted", "Not added to the locked 25-feature contract to avoid changing the model."],
];
const auditHeaders = ["locked_feature_or_group", "provenance", "PET UV-C aged", "PE natural aging", "decision / limitation"];
audit.getRangeByIndexes(0, 0, auditRows.length + 1, auditHeaders.length).values = [auditHeaders, ...auditRows];
audit.tables.add(`A1:E${auditRows.length + 1}`, true, "ExternalFeatureMappingAuditTable");
audit.showGridLines = false;
audit.freezePanes.freezeRows(1);
audit.getRange("A1:E1").format = {fill: "#0F766E", font: {bold: true, color: "#FFFFFF"}, wrapText: true, horizontalAlignment: "center"};
audit.getRange("A:E").format.wrapText = true;
audit.getRange("A:A").format.columnWidth = 30;
audit.getRange("B:B").format.columnWidth = 35;
audit.getRange("C:D").format.columnWidth = 28;
audit.getRange("E:E").format.columnWidth = 65;
audit.getRange(`A2:E${auditRows.length + 1}`).format.rowHeight = 38;

const expected = rawFeatures.length;
for (const row of modelRows) {
  const values = Object.fromEntries(rawFeatures.map((feature, i) => [feature, row[4 + i]]));
  const metalCount = values.metal_cd + values.metal_cr + values.metal_hg;
  const polymerCount = values.ret_other + values.ret_pa + values.ret_pe + values.ret_pet + values.ret_pla + values.ret_pp + values.ret_ps + values.ret_pvc;
  const agingCount = values.ags_aged + values.ags_virgin;
  if (metalCount !== 1 || polymerCount !== 1 || agingCount !== 1) throw new Error(`Invalid one-hot mapping for ${row[0]}`);
}
if (rawFeatures.length !== expected || modelRows.length !== 35) throw new Error("Schema or row-count validation failed.");

console.log((await wb.inspect({kind: "workbook,sheet,table", maxChars: 3000, tableMaxRows: 3, tableMaxCols: 10})).ndjson);
if (previewPath) {
  const preview = await wb.render({sheetName: "Model_Input_25Features", range: "A1:AE12", scale: 1, format: "png"});
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(`Wrote ${outputPath}`);
