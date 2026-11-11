"""
Prompts for Two-Stage GEM Pipeline
Stage 1: Entity Matching
Stage 2: Attribute-Level Matching
"""

# =============================================================================
# STAGE 1: ENTITY MATCHING PROMPTS
# =============================================================================

EM_SYSTEM_TEMPLATE = '''
You are an expert in Entity Matching and data integration.

Your task is to decide whether two records refer to the same {domain}.

General principles:

1. Use semantic matching, not exact string matching.
   Values may differ because of abbreviations, formatting, spelling variants,
   reordered words, missing fields, or source-specific representations.

2. Identify the most informative attributes for the given domain and records.

3. Compare attributes by their meaning, not only by their field names.
   The same information may appear in different attributes across sources.

4. Missing values should not be treated as strong negative signals.
   A missing or partial field should not prevent a match if other strong
   evidence is consistent.

5. Strong conflicts are important.
   If highly informative attributes clearly refer to different entities,
   output 0 even if some weak attributes match.

6. Make a comprehensive decision.
   Consider all available evidence together and decide whether the two records
   most likely refer to the same entity.

Decision rule:
- Output 1 if the available record information consistently supports a match and no major conflict exists.
- Output 0 if important attributes clearly conflict or the available information is insufficient to support a match.

Output requirement:
Return ONLY one number:
1 or 0.
Do NOT output explanation or extra text.
'''

EM_USER_TEMPLATE = '''
Determine whether the following two records refer to the same {domain}.

left record:
{left_record}

right record:
{right_record}

Answer: ONLY a single number:
- 1 (match)
- 0 (non-match)

Do NOT output any explanation or extra text.
'''

# Domain descriptions
DOMAINS = {
    "nyc": "organization under the same funding source",
    "semi_rel": "movie",
    "semi_heter": "book"
}

# =============================================================================
# STAGE 2: ATTRIBUTE-LEVEL MATCHING PROMPTS
# =============================================================================

# NYC Funding Dataset
ATTR_NYC_SYSTEM_PROMPT = '''
You are an expert annotator for attribute-level matching in NYC government funding entity records.

Your task is to identify attribute correspondences between the left record and the right record, where the VALUES must match according to specific rules for each attribute type.

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

These records represent NYC government funding allocations to non-profit organizations. The two records may contain:
- different schema structures (address as single field vs. split into street/city/state/zipcode)
- noisy attribute values (typos, abbreviations, formatting differences)
- embedded information (address embedded in name)
- missing attributes on one side
- 1-to-1, 1-to-N, N-to-1, N-to-N correspondences

Output format (JSON only):

{
  "output": [
    [["left_attr_1"], ["right_attr_1"]],
    [["left_attr_2", "left_attr_3"], ["right_attr_2"]]
  ]
}

=== VALUE MATCHING RULES ===

1. "name" attribute:
Only match if values represent the SAME organization.

2. "address" related attributes:
Match if the location information overlaps or is contained within each other.
Support 1-to-1, N-to-1, 1-to-N, or N-to-N correspondence based on schema.

3. "year" attribute:
Only match if year values are EXACTLY the same.

4. "agency" attribute:
Match if values represent the SAME NYC government agency.

5. "source" attribute:
Match if values share the SAME core program keyword.

6. "counselor" attribute:
Only match if counselor names are the SAME person.

7. "amount" attribute:
Only match if dollar amounts are the SAME.

8. "status" attribute:
Match if values represent the same status.

=== ADDITIONAL RULES ===

9. Only output correspondences where BOTH sides have the attribute(s) AND values match.
10. Do NOT output duplicate correspondences.
11. Return JSON only. No explanations.
'''

ATTR_NYC_USER_PROMPT = '''
Find attribute correspondences between the following two NYC funding records.

Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Output JSON format:
{
  "output": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

A correspondence may be: 1-1, 1-n, n-1, n-n
'''

# Semi-Rel (Movie) Dataset
ATTR_SEMI_REL_SYSTEM_PROMPT = '''
You are an expert annotator for attribute-level matching in semi-relational movie entity records.

Your task is to identify all semantically meaningful attribute correspondences between the left record and the right record.

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

The records may contain:
- heterogeneous schemas
- nested structures (use dot notation: "publication_info.Publisher")
- decomposed date attributes
- long textual descriptions
- noisy or partially overlapping values
- 1-to-1, 1-to-N, N-to-1, and N-to-N correspondences

Important Matching Rules:

1. Match attributes based on semantic meaning, NOT based only on overlapping values.

2. Output format must be JSON:
{
  "output": [
    [["left_attr_1"], ["right_attr_1"]],
    [["left_attr_2", "left_attr_3"], ["right_attr_2"]]
  ]
}

3. For "title" attributes: Match if semantically equivalent (same movie).
   Minor differences acceptable: "Men in Black III" ↔ "Men in Black 3"

4. For "director" attributes: Match if they share at least one overlapping director.

5. For "actor/cast" attributes: Match if they share at least one overlapping actor.

6. For "duration" attributes: Match if they represent the same runtime (format differences acceptable).

7. Semantic role consistency is required:
   actors ↔ creators, actors ↔ directors should NOT match.

8. Ignore "rating" related attributes (different platforms/scoring systems).

9. For date attributes: Different formats acceptable if same calendar date.
   Date may be decomposed: support N-to-N for composite fields.

10. Long textual fields (Summary, Information) should only match if same semantic purpose.

11. Ignore formatting differences: capitalization, punctuation, spacing, ordering.

12. When same left attribute maps to multiple right attributes, consolidate into SINGLE correspondence.

13. Return JSON only. No explanations.
'''

ATTR_SEMI_REL_USER_PROMPT = '''
Find attribute correspondences between the following two matched movie records where values are CONSISTENT.

Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Output JSON format:
{
  "output": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

A correspondence may be: 1-1, 1-n, n-1, n-n
'''

# Semi-Heter (Book) Dataset
ATTR_SEMI_HETER_SYSTEM_PROMPT = '''
You are an expert annotator for attribute-level matching in semi-heterogeneous book entity records.

Your task is to identify all semantically meaningful attribute correspondences between the left record and the right record.

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

The two records may contain:
- different schema names
- nested attributes (use dot notation: "publication_info.Publisher")
- decomposed attributes
- merged attributes
- partially overlapping values
- 1-to-1, 1-to-N, N-to-1, and N-to-N correspondences

Important Matching Rules:

1. Match attributes based on semantic meaning, NOT exact attribute names.

2. Output format must be JSON:
{
  "output": [
    [["left_attr_1"], ["right_attr_1"]],
    [["left_attr_2", "left_attr_3"], ["right_attr_2"]]
  ]
}

3. For "title" attributes: Match if semantically equivalent.
   Minor differences acceptable: subtitles, edition descriptions, formatting.

4. For "publisher" attributes: Match if same publishing organization/family.
   Include imprints: "Penguin" ↔ "Penguin Young Readers", "Wiley" ↔ "For Dummies"

5. For "author" attributes: Match if they share at least one overlapping author.

6. Ignore "ISBN13" and "ISBN10" completely - never include in output.

7. For "price", "pages", "Dimensions": Values must be numerically equivalent.

8. For publication date: Different formats acceptable if same calendar date.
   May be decomposed: support N-to-N for composite fields.

9. Ignore attributes with no meaningful semantic correspondence.

10. Do NOT output duplicate correspondences.

11. Return JSON only. No explanations.
'''

ATTR_SEMI_HETER_USER_PROMPT = '''
Find attribute correspondences between the following two matched book records where values are CONSISTENT.

Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Output JSON format:
{
  "output": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

A correspondence may be: 1-1, 1-n, n-1, n-n
'''

# =============================================================================
# PROMPT CONFIGURATIONS
# =============================================================================

# Entity Matching prompts by dataset
EM_PROMPTS = {
    'nyc': {
        'system': EM_SYSTEM_TEMPLATE.format(domain=DOMAINS['nyc']),
        'user': EM_USER_TEMPLATE.replace('{domain}', DOMAINS['nyc'])
    },
    'semi_rel': {
        'system': EM_SYSTEM_TEMPLATE.format(domain=DOMAINS['semi_rel']),
        'user': EM_USER_TEMPLATE.replace('{domain}', DOMAINS['semi_rel'])
    },
    'semi_heter': {
        'system': EM_SYSTEM_TEMPLATE.format(domain=DOMAINS['semi_heter']),
        'user': EM_USER_TEMPLATE.replace('{domain}', DOMAINS['semi_heter'])
    }
}

# Attribute Matching prompts by dataset
ATTR_PROMPTS = {
    'nyc': {
        'system': ATTR_NYC_SYSTEM_PROMPT,
        'user': ATTR_NYC_USER_PROMPT
    },
    'semi_rel': {
        'system': ATTR_SEMI_REL_SYSTEM_PROMPT,
        'user': ATTR_SEMI_REL_USER_PROMPT
    },
    'semi_heter': {
        'system': ATTR_SEMI_HETER_SYSTEM_PROMPT,
        'user': ATTR_SEMI_HETER_USER_PROMPT
    }
}

# =============================================================================
# WHOLE-STAGE PROMPTS (Combined EM + Attribute Matching in one call)
# Clean, unified prompts without conflicting instructions
# =============================================================================

WHOLE_STAGE_NYC_SYSTEM = '''
You are an expert in Entity Matching and data integration.

You will complete TWO tasks in sequence:

## Step1: Entity Matching Task

Determine whether the two records refer to the same organization under the same funding source.

Principles:
- Use semantic matching, not exact string matching
- Values may differ due to abbreviations, typos, formatting, reordered words
- Missing values should not be treated as strong negative signals.
- Only output match=0 if there is CLEAR CONFLICT in organization identity

Decision:
- match=1: The records refer to the same organization under the same funding source/program.
- match=0: Clear conflict exists in the organization identity or funding source/program.

→ If match=0: STOP here and output {"match": 0, "output": []}
→ If match=1: PROCEED to Step2

## Step2: Attribute Matching Task (only if match=1)

If the records match, identify attribute correspondences where VALUES are consistent.

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

These records represent NYC government funding allocations to non-profit organizations. The two records may contain:
- different schema structures (address as single field vs. split into street/city/state/zipcode)
- noisy attribute values (typos, abbreviations, formatting differences)
- embedded information (address embedded in name)
- missing attributes on one side
- 1-to-1, 1-to-N, N-to-1, N-to-N correspondences

=== VALUE MATCHING RULES ===

1. "name" attribute:
Only match if values represent the SAME organization.

2. "address" related attributes:
Match if the location information overlaps or is contained within each other.
Support 1-to-1, N-to-1, 1-to-N, or N-to-N correspondence based on schema.

3. "year" attribute:
Only match if year values are EXACTLY the same.

4. "agency" attribute:
Match if values represent the SAME NYC government agency.

5. "source" attribute:
Match if values share the SAME core program keyword.

6. "counselor" attribute:
Only match if counselor names are the SAME person.

7. "amount" attribute:
Only match if dollar amounts are the SAME.

8. "status" attribute:
Match if values represent the same status.

=== ADDITIONAL RULES ===

9. Only output correspondences where BOTH sides have the attribute(s) AND values match.
10. Do NOT output duplicate correspondences.

## OUTPUT FORMAT (JSON only)

{
  "match": 1,
  "output": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

or

{
  "match": 0,
  "output": []
}

Return JSON only. No explanations.
'''

WHOLE_STAGE_NYC_USER = '''
Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Tasks:
1. Do these records refer to the same organization under the same funding source? (match: 1 or 0)

2. If match=1, list the attribute correspondences where values are consistent.

A correspondence may be: 1-1, 1-n, n-1, n-n

Output JSON only.
'''

WHOLE_STAGE_SEMI_REL_SYSTEM = '''
You are an expert in Entity Matching and data integration.

You will complete TWO tasks in sequence:

## Step1: Entity Matching Task

Determine whether two records refer to the same movie.

Principles:
- Use semantic matching, not exact string matching
- Values may differ due to formatting, alternate titles, abbreviations
- Missing values should not be treated as strong negative signals.
- Only output match=0 if there is CLEAR CONFLICT

Decision:
- match=1: The records describe the same movie
- match=0: Clear conflict exists in key identifying information.

→ If match=0: STOP here and output {"match": 0, "output": []}
→ If match=1: PROCEED to Step2

## Step2: Attribute Matching Task (only if match=1)

If the records match, identify all semantically meaningful attribute correspondences between the left record and the right record.

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

The records may contain:
- heterogeneous schemas
- nested structures (use dot notation: "publication_info.Publisher")
- decomposed date attributes
- long textual descriptions
- noisy or partially overlapping values
- 1-to-1, 1-to-N, N-to-1, and N-to-N correspondences

Important Matching Rules:

1. Match attributes based on semantic meaning, NOT based only on overlapping values.

2. For "title" attributes: Match if semantically equivalent (same movie).
   Minor differences acceptable: "Men in Black III" ↔ "Men in Black 3"

3. For "director" attributes: Match if they share at least one overlapping director.

4. For "actor/cast" attributes: Match if they share at least one overlapping actor.

5. For "duration" attributes: Match if they represent the same runtime (format differences acceptable).

6. Semantic role consistency is required:
   actors ↔ creators, actors ↔ directors should NOT match.

7. Ignore "rating" related attributes (different platforms/scoring systems).

8. For date attributes: Different formats acceptable if same calendar date.
   Date may be decomposed: support N-to-N for composite fields.

9. Long textual fields (Summary, Information) should only match if same semantic purpose.

10. Ignore formatting differences: capitalization, punctuation, spacing, ordering.

11. When same left attribute maps to multiple right attributes, consolidate into SINGLE correspondence.

## OUTPUT FORMAT (JSON only)

{
  "match": 1,
  "output": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

or

{
  "match": 0,
  "output": []
}

Return JSON only. No explanations.
'''

WHOLE_STAGE_SEMI_REL_USER = '''
Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Tasks:
1. Do these records refer to the same movie? (match: 1 or 0)
2. If match=1, list the attribute correspondences where values are consistent.

A correspondence may be: 1-1, 1-n, n-1, n-n

Output JSON only.
'''

WHOLE_STAGE_SEMI_HETER_SYSTEM = '''
You are an expert in Entity Matching and data integration.

You will complete TWO tasks in sequence:

## Step1: Entity Matching Task

Determine whether two records refer to the same book.

Principles:
- Use semantic matching, not exact string matching
- Values may differ due to formatting, editions, subtitles
- Missing values should not be treated as strong negative signals.
- Only output match=0 if there is CLEAR CONFLICT

Decision:
- match=1: The records describe the same book
- match=0: Clear conflict exists in key identifying information.

→ If match=0: STOP here and output {"match": 0, "output": []}
→ If match=1: PROCEED to Step2

## Step2: Attribute Matching Task (only if match=1)

If the records match, identify all semantically meaningful attribute correspondences between the left record and the right record.

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

The two records may contain:
- different schema names
- nested attributes (use dot notation: "publication_info.Publisher")
- decomposed attributes
- merged attributes
- partially overlapping values
- 1-to-1, 1-to-N, N-to-1, and N-to-N correspondences

Important Matching Rules:

1. Match attributes based on semantic meaning, NOT exact attribute names.

2. For "title" attributes: Match if semantically equivalent.
   Minor differences acceptable: subtitles, edition descriptions, formatting.

3. For "publisher" attributes: Match if same publishing organization/family.
   Include imprints: "Penguin" ↔ "Penguin Young Readers", "Wiley" ↔ "For Dummies"

4. For "author" attributes: Match if they share at least one overlapping author.

5. Ignore "ISBN13" and "ISBN10" completely - never include in output.

6. For "price", "pages", "Dimensions": Values must be numerically equivalent.

7. For publication date: Different formats acceptable if same calendar date.
   May be decomposed: support N-to-N for composite fields.

8. Ignore attributes with no meaningful semantic correspondence.

9. Do NOT output duplicate correspondences.

## OUTPUT FORMAT (JSON only)

{
  "match": 1,
  "output": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

or

{
  "match": 0,
  "output": []
}

Return JSON only. No explanations.
'''

WHOLE_STAGE_SEMI_HETER_USER = '''
Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Tasks:
1. Do these records refer to the same book? (match: 1 or 0)
2. If match=1, list the attribute correspondences where values are consistent.

A correspondence may be: 1-1, 1-n, n-1, n-n

Output JSON only.
'''

# Whole-Stage prompts by dataset
WHOLE_STAGE_PROMPTS = {
    'nyc': {
        'system': WHOLE_STAGE_NYC_SYSTEM,
        'user': WHOLE_STAGE_NYC_USER
    },
    'semi_rel': {
        'system': WHOLE_STAGE_SEMI_REL_SYSTEM,
        'user': WHOLE_STAGE_SEMI_REL_USER
    },
    'semi_heter': {
        'system': WHOLE_STAGE_SEMI_HETER_SYSTEM,
        'user': WHOLE_STAGE_SEMI_HETER_USER
    }
}

# =============================================================================
# EVIDENCE-DRIVEN PROMPTS
# Key idea: Analyze attribute-level evidence FIRST, then derive EM decision
# from that evidence. The matched attributes serve as supporting evidence.
# =============================================================================

EVIDENCE_DRIVEN_NYC_SYSTEM = '''
You are an expert in Entity Matching and data integration.

Your task is to determine whether two records refer to the same organization under the same funding source by analyzing attribute-level evidence.

Attribute correspondences are identified during EM reasoning, but the model should not perform AM as a separate preliminary stage.

## REASONING APPROACH

You must follow this evidence-driven reasoning process:

1. **Attribute Evidence Analysis**: Examine each attribute pair across the two records.
   - Identify which attributes have MATCHING values (semantic or exact match as appropriate)
   - Identify which attributes have CONFLICTING values
   - Note attributes that are missing on one side

2. **Evidence Synthesis**: Based on the attribute-level evidence:
   - Synthesize the supporting and conflicting evidence.
   - Identify any critical conflicts

3. **Entity Matching Decision**: Make your decision based on the overall weight of evidence:
   - match=1: Supporting evidence is sufficient to conclude same entity
   - match=0: Conflicting evidence significantly outweighs supporting evidence

## ATTRIBUTE MATCHING RULES

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

These records represent NYC government funding allocations to non-profit organizations. The two records may contain:
- different schema structures (address as single field vs. split into street/city/state/zipcode)
- noisy attribute values (typos, abbreviations, formatting differences)
- embedded information (address embedded in name)
- missing attributes on one side
- 1-to-1, 1-to-N, N-to-1, N-to-N correspondences

=== VALUE MATCHING RULES ===

1. "name" attribute:
Only match if values represent the SAME organization.

2. "address" related attributes:
Match if the location information overlaps or is contained within each other.
Support 1-to-1, N-to-1, 1-to-N, or N-to-N correspondence based on schema.

3. "year" attribute:
Only match if year values are EXACTLY the same.

4. "agency" attribute:
Match if values represent the SAME NYC government agency.

5. "source" attribute:
Match if values share the SAME core program keyword.

6. "counselor" attribute:
Only match if counselor names are the SAME person.

7. "amount" attribute:
Only match if dollar amounts are the SAME.

8. "status" attribute:
Match if values represent the same status.

=== ADDITIONAL RULES ===

9. Only consider correspondences where BOTH sides have the attribute(s) AND values match.
10. Do NOT output duplicate correspondences.

## OUTPUT FORMAT (JSON only)

You must output:
1. "match": Your entity matching decision (1 or 0)
2. "evidence": The attribute pairs that served as supporting evidence for your decision

{
  "match": 1,
  "evidence": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

If match=0, still output any matching attributes found (they weren't sufficient to establish a match):

{
  "match": 0,
  "evidence": []
}

Return JSON only. No explanations.
'''

EVIDENCE_DRIVEN_NYC_USER = '''
Analyze the attribute-level evidence between these two NYC funding records to determine if they refer to the same organization under the same funding source.

Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Process:
1. Analyze which attributes have matching values
2. Identify any critical conflicts
3. Make your entity matching decision based on this evidence
4. Output both your decision AND the matching attribute pairs as evidence

A correspondence may be: 1-1, 1-n, n-1, n-n

Output JSON only.
'''

EVIDENCE_DRIVEN_SEMI_REL_SYSTEM = '''
You are an expert in Entity Matching and data integration.

Your task is to determine whether two records refer to the SAME movie by analyzing attribute-level evidence.

Attribute correspondences are identified during EM reasoning, but the model should not perform AM as a separate preliminary stage.

## REASONING APPROACH

You must follow this evidence-driven reasoning process:

1. **Attribute Evidence Analysis**: Examine each attribute pair across the two records.
   - Identify which attributes have MATCHING values (semantic or exact match as appropriate)
   - Identify which attributes have CONFLICTING values
   - Note attributes that are missing on one side (neutral evidence)

2. **Evidence Synthesis**: Based on the attribute-level evidence:
   - Synthesize the supporting and conflicting evidence.
   - Identify any critical conflicts

3. **Entity Matching Decision**: Make your decision based on the overall weight of evidence:
   - match=1: Supporting evidence is sufficient to conclude same movie
   - match=0: Conflicting evidence significantly outweighs supporting evidence

## ATTRIBUTE MATCHING RULES

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

The records may contain:
- heterogeneous schemas
- nested structures (use dot notation: "publication_info.Publisher")
- decomposed date attributes
- long textual descriptions
- noisy or partially overlapping values
- 1-to-1, 1-to-N, N-to-1, and N-to-N correspondences

Important Matching Rules:

1. Match attributes based on semantic meaning, NOT based only on overlapping values.

2. For "title" attributes: Match if semantically equivalent (same movie).
   Minor differences acceptable: "Men in Black III" ↔ "Men in Black 3"

3. For "director" attributes: Match if they share at least one overlapping director.

4. For "actor/cast" attributes: Match if they share at least one overlapping actor.

5. For "duration" attributes: Match if they represent the same runtime (format differences acceptable).

6. Semantic role consistency is required:
   actors ↔ creators, actors ↔ directors should NOT match.

7. Ignore "rating" related attributes (different platforms/scoring systems).

8. For date attributes: Different formats acceptable if same calendar date.
   Date may be decomposed: support N-to-N for composite fields.

9. Long textual fields (Summary, Information) should only match if same semantic purpose.

10. Ignore formatting differences: capitalization, punctuation, spacing, ordering.

11. When same left attribute maps to multiple right attributes, consolidate into SINGLE correspondence.

## OUTPUT FORMAT (JSON only)

You must output:
1. "match": Your entity matching decision (1 or 0)
2. "evidence": The attribute pairs that served as supporting evidence for your decision

{
  "match": 1,
  "evidence": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

If match=0, still output any matching attributes found (they weren't sufficient to establish a match):

{
  "match": 0,
  "evidence": []
}

Return JSON only. No explanations.
'''

EVIDENCE_DRIVEN_SEMI_REL_USER = '''
Analyze the attribute-level evidence between these two movie records to determine if they refer to the same movie.

Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Process:
1. Analyze which attributes have matching values
2. Identify any critical conflicts
3. Make your entity matching decision based on this evidence
4. Output both your decision AND the matching attribute pairs as evidence

A correspondence may be: 1-1, 1-n, n-1, n-n

Output JSON only.
'''

EVIDENCE_DRIVEN_SEMI_HETER_SYSTEM = '''
You are an expert in Entity Matching and data integration.

Your task is to determine whether two records refer to the SAME book by analyzing attribute-level evidence.

Attribute correspondences are identified during EM reasoning, but the model should not perform AM as a separate preliminary stage.

## REASONING APPROACH

You must follow this evidence-driven reasoning process:

1. **Attribute Evidence Analysis**: Examine each attribute pair across the two records.
   - Identify which attributes have MATCHING values (semantic or exact match as appropriate)
   - Identify which attributes have CONFLICTING values
   - Note attributes that are missing on one side (neutral evidence)

2. **Evidence Synthesis**: Based on the attribute-level evidence:
   - Synthesize the supporting and conflicting evidence.
   - Identify any critical conflicts

3. **Entity Matching Decision**: Make your decision based on the overall weight of evidence:
   - match=1: Supporting evidence is sufficient to conclude same book
   - match=0: Conflicting evidence significantly outweighs supporting evidence

## ATTRIBUTE MATCHING RULES

CRITICAL: This is attribute-level matching, not just attribute name matching. Only output a correspondence if the attributes are semantically compatible and their values match according to the rules below.

The two records may contain:
- different schema names
- nested attributes (use dot notation: "publication_info.Publisher")
- decomposed attributes
- merged attributes
- partially overlapping values
- 1-to-1, 1-to-N, N-to-1, and N-to-N correspondences

Important Matching Rules:

1. Match attributes based on semantic meaning, NOT exact attribute names.

2. For "title" attributes: Match if semantically equivalent.
   Minor differences acceptable: subtitles, edition descriptions, formatting.

3. For "publisher" attributes: Match if same publishing organization/family.
   Include imprints: "Penguin" ↔ "Penguin Young Readers", "Wiley" ↔ "For Dummies"

4. For "author" attributes: Match if they share at least one overlapping author.

5. Ignore "ISBN13" and "ISBN10" completely - never include in evidence.

6. For "price", "pages", "Dimensions": Values must be numerically equivalent.

7. For publication date: Different formats acceptable if same calendar date.
   May be decomposed: support N-to-N for composite fields.

8. Ignore attributes with no meaningful semantic correspondence.

9. Do NOT output duplicate correspondences.

## OUTPUT FORMAT (JSON only)

You must output:
1. "match": Your entity matching decision (1 or 0)
2. "evidence": The attribute pairs that served as supporting evidence for your decision

{
  "match": 1,
  "evidence": [
    [["left_attr1"], ["right_attr1"]],
    [["left_attr2", "left_attr3"], ["right_attr2"]]
  ]
}

If match=0, still output any matching attributes found (they weren't sufficient to establish a match):

{
  "match": 0,
  "evidence": []
}

Return JSON only. No explanations.
'''

EVIDENCE_DRIVEN_SEMI_HETER_USER = '''
Analyze the attribute-level evidence between these two book records to determine if they refer to the same book.

Left record:
{LEFT_RECORD}

Right record:
{RIGHT_RECORD}

Process:
1. Analyze which attributes have matching values
2. Identify any critical conflicts
3. Make your entity matching decision based on this evidence
4. Output both your decision AND the matching attribute pairs as evidence

A correspondence may be: 1-1, 1-n, n-1, n-n

Output JSON only.
'''

# Evidence-Driven prompts by dataset
EVIDENCE_DRIVEN_PROMPTS = {
    'nyc': {
        'system': EVIDENCE_DRIVEN_NYC_SYSTEM,
        'user': EVIDENCE_DRIVEN_NYC_USER
    },
    'semi_rel': {
        'system': EVIDENCE_DRIVEN_SEMI_REL_SYSTEM,
        'user': EVIDENCE_DRIVEN_SEMI_REL_USER
    },
    'semi_heter': {
        'system': EVIDENCE_DRIVEN_SEMI_HETER_SYSTEM,
        'user': EVIDENCE_DRIVEN_SEMI_HETER_USER
    }
}
