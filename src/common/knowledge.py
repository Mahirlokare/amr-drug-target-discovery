"""
Scientific AI Knowledge & Drug Discovery Intelligence Engine.
Provides comprehensive, domain-grounded answers for AMR mechanisms, targets,
mutations, chemical pharmacophores, ADMET interpretation, and screening algorithms.
"""

from typing import Any, Dict, List, Optional
import re

from src.data.resolver import chemical_resolver, DRUG_NAME_DICTIONARY

AMR_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "topic": "InhA Target & Mechanism",
        "keywords": ["inha", "rv1484", "fas-ii", "enoyl-acp", "mycolic acid", "isoniazid", "mechanism", "catalytic"],
        "title": "Mycobacterium tuberculosis InhA (Rv1484) Catalytic Mechanism",
        "content": (
            "InhA is an NADH-dependent enoyl-[acyl-carrier-protein] reductase belonging to the FAS-II (fatty acid synthase II) pathway. "
            "It catalyzes the stereospecific reduction of 2-trans-enoyl-ACP to acyl-ACP, an essential rate-limiting step in the synthesis "
            "of very long-chain (C60–C90) mycolic acids that constitute the impermeable lipid envelope of M. tuberculosis.\n\n"
            "• **Catalytic Residues**: Tyr158 acts as the catalytic acid; Phe149 and Met155 form hydrophobic stacking walls for the acyl chain; "
            "Lys165 coordinates the pyrophosphate of the NADH cofactor.\n"
            "• **Clinical Repertoire**: Direct InhA inhibitors (such as NITD-916 and GEQ series) bypass the prodrug activation pathway of KatG, "
            "retaining full potency against KatG-mutant MDR-TB clinical isolates."
        ),
        "references": ["PDB: 4TZK", "UniProt: P9WGR1", "Banerjee et al., Science 1994", "Vilchèze & Jacobs, Microbiol Spectr 2014"]
    },
    {
        "topic": "InhA Resistance Mutations",
        "keywords": ["s94a", "i21v", "i16t", "i47t", "mutation", "resistance", "mutant"],
        "title": "InhA Structural Resistance Mutations & Escape Profiles",
        "content": (
            "Target-derived mutations in the inhA structural gene cause high-level cross-resistance to both isoniazid and ethionamide:\n\n"
            "1. **S94A**: The canonical substitution. Ser94 normally donates a hydrogen bond to the 2'-hydroxyl group of the NADH ribose. "
            "Alanine disruption weakens NADH and INH-NAD adduct binding affinity >5-fold without abolishing basal enzymatic turnover.\n"
            "2. **I21V**: Located in the hydrophobic pocket flanking the nicotinamide ring; accelerates adduct dissociation.\n"
            "3. **I16T & I47T**: Subtle steric perturbations in the substrate-binding entrance tunnel and cofactor loop.\n"
            "4. **Promoter mutations (c-15t)**: Cause transcriptional overexpression of wild-type InhA, titrating out drug concentration."
        ),
        "references": ["Dias et al., J Biol Chem 2007", "Dessen et al., Science 1995"]
    },
    {
        "topic": "DprE1 Target & Inhibitors",
        "keywords": ["dpre1", "rv3790", "epimerase", "macozinone", "pbtz169", "btz043", "arabinogalactan"],
        "title": "M. tuberculosis DprE1 (Decaprenylphosphoryl-beta-D-ribose 2'-epimerase)",
        "content": (
            "DprE1 (Rv3790) is an essential flavoprotein (FAD cofactor) epimerase. In tandem with DprE2, it converts decaprenylphosphoryl-D-ribose (DPR) "
            "into decaprenylphosphoryl-D-arabinose (DPA), the sole precursor for arabinan biosynthesis in the mycobacterial arabinogalactan/lipoarabinomannan cell wall.\n\n"
            "• **Covalent Inhibitors**: Benzothiazinones (Macozinone / PBTZ169, BTZ043) undergo nitroreduction by FADH2 to form a reactive nitroso intermediate "
            "that establishes a covalent semimercaptal bond with catalytic residue Cys387.\n"
            "• **Resistance Mutation C387S**: Substitutes cysteine for serine, abolishing covalent adduct formation."
        ),
        "references": ["PDB: 4KW5", "Makarov et al., Science 2009", "Batt et al., ACS Chem Biol 2012"]
    },
    {
        "topic": "DNA Gyrase GyrA (MRSA)",
        "keywords": ["gyra", "gyrase", "staphylococcus", "mrsa", "fluoroquinolone", "ciprofloxacin", "s84l"],
        "title": "S. aureus DNA Gyrase A (GyrA) & Fluoroquinolone Resistance",
        "content": (
            "GyrA forms a heterotetrameric (A2B2) complex responsible for introducing negative supercoils into double-stranded bacterial DNA during replication.\n\n"
            "• **Inhibition Mechanism**: Fluoroquinolones (ciprofloxacin, moxifloxacin) stabilize covalent enzyme-cleaved DNA binary complexes, generating fatal double-strand breaks.\n"
            "• **QRDR Resistance (S84L / E88K)**: Ser84 and Glu88 coordinate a hydrated Mg2+ ion bridge to the drug keto-acid group. Point mutations dismantle this bridge, causing 16- to 64-fold MIC elevation."
        ),
        "references": ["PDB: 2XCS", "Drlica et al., Curr Mol Med 2008"]
    },
    {
        "topic": "NDM-1 Carbapenemase",
        "keywords": ["ndm", "ndm-1", "carbapenemase", "metallo-beta-lactamase", "meropenem", "carbapenem", "zinc"],
        "title": "New Delhi Metallo-Beta-Lactamase 1 (NDM-1)",
        "content": (
            "NDM-1 is a Class B1 zinc metallo-beta-lactamase capable of hydrolyzing penicillin, cephalosporin, and carbapenem ring structures.\n\n"
            "• **Bimetallic Catalytic Active Site**: Houses two Zn2+ ions coordinated by His120, His122, Asp124, His189, Cys208, and His250, polarizing a bridging hydroxide ion that nucleophilically attacks the beta-lactam carbonyl.\n"
            "• **Therapeutic Challenge**: Classic beta-lactamase inhibitors (clavulanate, sulbactam, tazobactam) do NOT inhibit NDM-1. Novel zinc chelators and boronic acid inhibitors represent current lead pipelines."
        ),
        "references": ["PDB: 3SPU", "Yong et al., J Antimicrob Chemother 2009", "King et al., Nature 2014"]
    },
    {
        "topic": "Scaffold Leakage & Bemis-Murcko Split",
        "keywords": ["leakage", "scaffold", "bemis-murcko", "generalization", "split", "random split"],
        "title": "Scaffold-Aware Validation & Zero-Leakage Protocol",
        "content": (
            "Random train/test splits severely overestimate machine learning generalization in medicinal chemistry because structurally near-identical analogs "
            "are distributed across both training and test sets (producing artificial ~0.95+ AUROC scores that collapse on novel series).\n\n"
            "• **Bemis-Murcko Protocol**: Strips all side chains to isolate core ring topologies and linkers. Entire scaffold clusters are assigned strictly "
            "to either Train (70%), Validation (15%), or Test (15%).\n"
            "• **Disjointness Assertion**: `set(train_scaffolds) ∩ set(test_scaffolds) == ∅`. Guarantees model evaluation reflects genuine out-of-series discovery."
        ),
        "references": ["Bemis & Murcko, J Med Chem 1996", "Project 9 Technical Architecture Blueprint Section 8.4"]
    },
    {
        "topic": "Epistemic Uncertainty & Deep Ensembles",
        "keywords": ["uncertainty", "epistemic", "ensemble", "variance", "std dev", "ood"],
        "title": "Epistemic Uncertainty Quantification via Deep Ensembles",
        "content": (
            "In high-stakes antimicrobial discovery, knowing when an AI model does NOT know is critical to avoid expensive false-positive synthesis.\n\n"
            "• **Ensemble Spread**: 3 diverse random-seeded GNN+PLM models evaluate the same candidate. Mean calibrated probability represents expected affinity; standard deviation represents epistemic disagreement.\n"
            "• **Triage Policy**: Molecules with uncertainty σ > 0.15 are flagged as `UNCERTAIN_PREDICTION`, prompting active-learning wet-lab confirmation before prioritization."
        ),
        "references": ["Lakshminarayanan et al., NeurIPS 2017", "Gawlikowski et al., J Artif Intell Res 2023"]
    },
    {
        "topic": "ADMET & Early Safety Filters",
        "keywords": ["admet", "lipinski", "veber", "pains", "qed", "sa", "synthetic accessibility", "solubility"],
        "title": "In-Silico ADMET Profiling & Liability Triage",
        "content": (
            "Calculated drug-likeness rules triage compounds that would otherwise fail downstream PK/PD:\n\n"
            "1. **Lipinski Rule of 5 (Ro5)**: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10. Predicts passive oral absorption.\n"
            "2. **Veber Rules**: Rotatable bonds ≤ 10, TPSA ≤ 140 Å². Correlates with membrane permeability.\n"
            "3. **PAINS (Pan-Assay Interference Compounds)**: 9 structural alert substructures (e.g. rhodanines, quinones, catechols) that non-specifically disrupt assay readouts.\n"
            "4. **QED (Quantitative Estimate of Drug-Likeness)**: Weighted desirability index from 0.0 (poor) to 1.0 (ideal drug).\n"
            "5. **Synthetic Accessibility (SA)**: Score from 1 (easy synthesis) to 10 (extremely difficult)."
        ),
        "references": ["Lipinski et al., Adv Drug Deliv Rev 2001", "Veber et al., J Med Chem 2002", "Baell & Holloway, J Med Chem 2010"]
    }
]


def answer_scientific_query(
    query_text: str,
    context_molecule: Optional[str] = None,
    context_target_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Intelligently analyzes and answers scientific questions regarding AMR targets,
    drug resistance, chemical properties, and discovery methodology.
    """
    clean_q = (query_text or "").strip().lower()
    if not clean_q:
        return {
            "title": "Scientific Discovery Assistant",
            "answer": "Ask any question regarding AMR target enzymes (InhA, DprE1, GyrA, PBP3, NDM-1), resistance mutations, drug candidates, or ADMET screening.",
            "references": [],
            "matched_topic": "General"
        }

    # Check if query contains any drug name from the dictionary
    for word in re.findall(r"[A-Za-z0-9\-]+", clean_q):
        if word in DRUG_NAME_DICTIONARY or word.replace("-", "") in DRUG_NAME_DICTIONARY:
            chem_res = chemical_resolver.resolve_input(word)
            if chem_res.get("valid") and chem_res.get("input_type") == "NAME_RESOLVED":
                name = chem_res.get("matched_name")
                return {
                    "title": f"Compound Profile: {name}",
                    "answer": (
                        f"**Chemical Information for {name}**:\n\n"
                        f"• **Canonical SMILES**: `{chem_res['canonical_smiles']}`\n"
                        f"• **Molecular Formula**: `{chem_res['formula']}`\n"
                        f"• **Molecular Weight**: `{chem_res['molecular_weight']} g/mol`\n"
                        f"• **Calculated LogP**: `{chem_res['logp']}`\n"
                        f"• **Topological Polar Surface Area (TPSA)**: `{chem_res['tpsa']} Å²`\n"
                        f"• **H-Bond Donors / Acceptors**: `{chem_res['hbd']} / {chem_res['hba']}`\n"
                        f"• **Rotatable Bonds**: `{chem_res['rotatable_bonds']}`\n\n"
                        f"*Tip: You can evaluate this compound directly in the screening simulator against any AMR target.*"
                    ),
                    "references": ["PubChem / ChEMBL Verified Therapeutic Benchmark"],
                    "matched_topic": "Chemical Entity"
                }

    # Match against AMR Knowledge Base
    best_match = None
    best_score = 0

    tokens = set(re.findall(r"\w+", clean_q))

    for item in AMR_KNOWLEDGE_BASE:
        score = sum(2 for kw in item["keywords"] if kw in clean_q)
        score += sum(1 for tok in tokens if any(tok in kw for kw in item["keywords"]))
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= 2:
        context_addon = ""
        if context_molecule:
            m_res = chemical_resolver.resolve_input(context_molecule)
            if m_res.get("valid"):
                context_addon = (
                    f"\n\n**Contextual Candidate Molecule**: `{m_res['canonical_smiles']}` "
                    f"(MW: {m_res['molecular_weight']}, LogP: {m_res['logp']})."
                )

        return {
            "title": best_match["title"],
            "answer": best_match["content"] + context_addon,
            "references": best_match["references"],
            "matched_topic": best_match["topic"]
        }

    # Generalized AI Scientific Fallback
    return {
        "title": "In-Silico AMR Research Intelligence",
        "answer": (
            f"You queried: *'{query_text}'*.\n\n"
            "This enterprise platform is configured for multi-pathogen antimicrobial resistance research:\n"
            "• **Primary Biological Targets**: *M. tuberculosis* InhA (Rv1484, FAS-II), DprE1 (Rv3790), *S. aureus* GyrA (MRSA), *P. aeruginosa* PBP3, and *K. pneumoniae* NDM-1.\n"
            "• **Computational Engine**: PyG Graph Convolutional Networks (GCN) + ESM-2 Protein Language Models (`esm2_t6_8M_UR50D`).\n"
            "• **Rigorous QC**: 0% Bemis-Murcko scaffold train/test leakage, 3-seed Deep Ensemble epistemic uncertainty, and 9 PAINS substructure liability filters.\n\n"
            "Try asking about: *'InhA S94A mutation'*, *'DprE1 mechanism'*, *'Macozinone'*, *'Scaffold leakage'*, *'ADMET filters'*, or enter any antibiotic name such as *'Isoniazid'*, *'Bedaquiline'*, or *'Ciprofloxacin'*."
        ),
        "references": ["WHO Bacterial Priority Pathogens List 2024", "Project 9 Technical Reference Guide"],
        "matched_topic": "General In-Silico Inquiry"
    }
