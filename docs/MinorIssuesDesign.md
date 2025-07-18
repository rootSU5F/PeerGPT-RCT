# RCT Minor Issues Detection Model Architecture

## Overview

The RCT Minor Issues Detection Model is a heuristic-based system designed to automatically identify and classify minor methodological issues in randomized controlled trial abstracts. The model follows the same architectural patterns as the major issues detection system, ensuring consistency and maintainability.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    RCT Minor Issues Detection Model             │
├─────────────────────────────────────────────────────────────────┤
│  Input Layer                                                    │
│  ├── Text Preprocessing                                         │
│  ├── spaCy NLP Pipeline                                         │
│  └── Input Validation                                           │
├─────────────────────────────────────────────────────────────────┤
│  Detection Engine                                               │
│  ├── Duration Follow-up Detector                               │
│  ├── Multicentre Study Detector                                │
│  ├── Primary Outcome Timeline Detector                         │
│  ├── Funding Source Detector                                   │
│  ├── Age Generalizability Detector                             │
│  ├── Sex Generalizability Detector                             │
│  └── Intention-to-Treat Detector                               │
├─────────────────────────────────────────────────────────────────┤
│  Processing Layer                                               │
│  ├── Pattern Matching Engine                                   │
│  ├── Context Validation                                        │
│  ├── Rule Application Logic                                    │
│  └── Result Aggregation                                        │
├─────────────────────────────────────────────────────────────────┤
│  Output Layer                                                  │
│  ├── Result Formatting                                         │
│  ├── CSV Export                                                │
│  ├── Batch Processing                                          │
│  └── Summary Statistics                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

### 1. Main Analysis Module

**Function**: `mainFunctionMinorIssues()`
- **Purpose**: Primary orchestration function
- **Input**: Abstract text, optional parameters
- **Output**: Dictionary of detection results
- **Responsibilities**:
  - Coordinate all detection functions
  - Aggregate results
  - Format output
  - Handle CSV export

### 2. Detection Functions Module

Each detector follows a standardized pattern:

```python
def check_[issue_type](text: str) -> str | None:
    # Input validation
    # Text preprocessing  
    # Pattern matching
    # Rule application
    # Return result or None
```

#### 2.1 Duration Follow-up Detector
- **Rule**: Detects follow-up < 1 year or not provided
- **Patterns**: Duration extraction, "not provided" indicators
- **Processing**: Unit conversion to months

#### 2.2 Multicentre Detector  
- **Rule**: Assumes single-center if not mentioned
- **Patterns**: Single/multi-center indicators
- **Processing**: Explicit mention detection

#### 2.3 Primary Outcome Timeline Detector
- **Rule**: Detects assessment ≤ 30 days or not mentioned
- **Patterns**: Timeline extraction, outcome context
- **Processing**: Unit conversion to days

#### 2.4 Funding Source Detector
- **Rule**: Identifies industry sponsorship
- **Patterns**: Company names, funding keywords
- **Processing**: Known pharma company matching

#### 2.5 Age Generalizability Detector
- **Rule**: Detects mean/median age < 70 years
- **Patterns**: Age statistics, ranges
- **Processing**: Age extraction and validation

#### 2.6 Sex Generalizability Detector
- **Rule**: Detects < 50% female participants
- **Patterns**: Percentages, count ratios
- **Processing**: Percentage calculation

#### 2.7 Intention-to-Treat Detector
- **Rule**: Detects missing ITT analysis
- **Patterns**: ITT keywords, analysis approaches
- **Processing**: Context-sensitive evaluation

### 3. Data Export Module

**Functions**: CSV export and batch processing
- **Single export**: `save_minor_results_to_csv()`
- **Batch export**: `analyze_multiple_abstracts_minor()`
- **Features**: Timestamp management, header handling

### 4. Utility Module

**Function**: `load_abstracts_from_file()`
- **Purpose**: File input handling
- **Formats**: Text files with various separators

## Technical Implementation

### Pattern Matching Engine

**Primary Technology**: Regular expressions with spaCy enhancement
- **Regex patterns**: Structured text extraction
- **spaCy processing**: Context understanding, entity recognition
- **Validation layers**: False positive reduction

### Rule Application Logic

Each detector implements specific business rules:

```python
# Example: Duration Follow-up Logic
if no_duration_patterns_found:
    return "Duration not provided message"
elif duration_in_months < 12:
    return "Short duration message"
else:
    return None  # No issue
```

### Context Validation

**Purpose**: Reduce false positives through contextual analysis
- **Exclusion patterns**: Filter irrelevant matches
- **Validation rules**: Ensure logical consistency
- **Range checking**: Verify realistic values

### Unit Conversion System

**Standardization**: Convert various units to common measures
- **Time units**: Days/weeks/months/years → months
- **Age ranges**: Calculate averages from ranges
- **Percentages**: Convert counts to percentages

## Data Flow Architecture

```
Input Abstract
      ↓
Text Preprocessing
├── Lowercase conversion
├── spaCy tokenization
└── Pattern preparation
      ↓
Pattern Matching
├── Regex extraction
├── Entity recognition
└── Context analysis
      ↓
Data Validation
├── Range checking
├── Unit conversion
└── Context filtering
      ↓
Rule Application
├── Threshold comparison
├── Business logic
└── Issue classification
      ↓
Result Aggregation
├── Collect all results
├── Count issues
└── Format messages
      ↓
Output Generation
├── Console display
├── CSV export
└── Summary statistics
```

## Error Handling Strategy

### Input Validation
- **Empty text**: Return standardized error message
- **Invalid format**: Graceful degradation
- **Encoding issues**: UTF-8 handling

### Processing Errors
- **Regex failures**: Continue with remaining patterns
- **Conversion errors**: Skip invalid values
- **spaCy errors**: Fallback to regex-only processing

### Output Consistency
- **Standardized messages**: Consistent format across detectors
- **Null handling**: Clear distinction between "no issue" and "no data"
- **Error propagation**: Maintain detection accuracy

## Performance Considerations

### Optimization Strategies
- **Pattern efficiency**: Optimized regex compilation
- **Early termination**: Stop processing when issue found
- **Batch processing**: Efficient multi-abstract handling

### Memory Management
- **spaCy model**: Single instance loading
- **Pattern caching**: Compiled regex reuse
- **Streaming**: Large file processing without memory overflow

### Scalability Features
- **Stateless design**: No dependencies between abstracts
- **Parallel processing**: Ready for multi-threading
- **Incremental output**: CSV append mode for large batches

## Quality Assurance

### Validation Framework
- **Test coverage**: All detection scenarios
- **Edge case handling**: Boundary conditions
- **Regression testing**: Consistent behavior

### Accuracy Metrics
- **Sensitivity**: Ability to detect actual issues
- **Specificity**: Avoid false positives
- **Rule compliance**: Adherence to clinical guidelines

### Maintenance Strategy
- **Modular design**: Independent detector functions
- **Pattern updates**: Easy rule modification
- **Version control**: Backward compatibility

## Integration Points

### Input Sources
- **Single abstracts**: Direct text input
- **CSV files**: Batch processing from dataframes
- **Text files**: File-based input with custom separators

### Output Formats
- **Console**: Human-readable immediate feedback
- **CSV**: Structured data for analysis
- **Return values**: Programmatic access to results

### External Dependencies
- **spaCy**: Advanced NLP processing
- **pandas**: Data manipulation for batch processing
- **re**: Pattern matching core functionality

## Configuration Management

### Detection Thresholds
- **Duration**: 12 months threshold
- **Timeline**: 30 days threshold  
- **Age**: 70 years threshold
- **Sex**: 50% female threshold

### Pattern Libraries
- **Company names**: Pharmaceutical industry database
- **Medical terms**: Clinical outcome vocabularies
- **Time expressions**: Comprehensive temporal patterns

### Rule Customization
- **Threshold adjustment**: Configurable limits
- **Pattern extension**: Additional detection rules
- **Context modification**: Refined validation logic

## Detection Function Specifications

### Function Output Matrix

| Function | Success Output | Error Output | No Issue Output |
|----------|----------------|--------------|-----------------|
| `check_duration_followup()` | Issue description string | "Empty text provided" | `None` |
| `check_multicentre()` | Issue description string | "Empty text provided" | `None` |
| `check_timeline_outcome()` | Issue description string | "Empty text provided" | `None` |
| `check_funding()` | Issue description string | "Empty text provided" | `None` |
| `check_generalizability_age()` | Issue description string | "Empty text provided" | `None` |
| `check_generalizability_sex()` | Issue description string | "Empty text provided" | `None` |
| `check_intention_to_treat()` | Issue description string | "Empty text provided" | `None` |

### Pattern Recognition Examples

#### Duration Follow-up Patterns
```regex
follow(?:\s|-)*up.*?(\d+(?:\.\d+)?)\s*(day|week|month|year)s?
duration.*?(?:not|no).*?(?:provided|specified|mentioned|reported)
```

#### Funding Source Patterns
```regex
funded\s+by\s+([A-Z][a-zA-Z\s&\-]+(?:Inc|Corp|Ltd|LLC|Pharmaceuticals?))
(pfizer|novartis|roche|merck|gsk|astrazeneca|sanofi|bayer)
```

#### Age Generalizability Patterns
```regex
(?:mean|median|average)\s+age.*?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)
age.*?(\d+)(?:\s|-)*(?:to|-)(?:\s|-)*(\d+)\s*(?:years?|yrs?)
```

#### Sex Generalizability Patterns
```regex
(\d+(?:\.\d+)?)\s*%\s*(?:were\s+)?(?:female|women)
(\d+)\s*(?:female|women).*?(\d+)\s*(?:male|men)
```

## API Reference

### Main Functions

```python
mainFunctionMinorIssues(abstract: str, save_to_csv: bool = True, csv_filename: str = None) -> Dict[str, Optional[str]]
```

```python
analyze_multiple_abstracts_minor(abstracts: List[str], csv_filename: str = "batch_minor_issues_analysis.csv") -> None
```

```python
save_minor_results_to_csv(abstract: str, results: Dict[str, Optional[str]], csv_filename: str = None) -> None
```

### Detection Functions

```python
check_duration_followup(text: str) -> str | None
check_multicentre(text: str) -> str | None
check_timeline_outcome(text: str) -> str | None
check_funding(text: str) -> str | None
check_generalizability_age(text: str) -> str | None
check_generalizability_sex(text: str) -> str | None
check_intention_to_treat(text: str) -> str | None
```

### Utility Functions

```python
load_abstracts_from_file(filename: str) -> List[str]
```