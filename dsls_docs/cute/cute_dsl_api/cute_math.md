# Math

The `cutlass.cute.math` module provides math functions for CuTe DSL programs.

## API documentation

*class* cutlass.cute.math.RoundingMode(*value*)
:   Bases: `str`, `Enum`

    IEEE 754 rounding modes for floating-point operations.

    NEAREST\_EVEN *= 'rn'*

    ZERO *= 'rz'*

    UP *= 'rp'*

    DOWN *= 'rm'*

cutlass.cute.math.sin( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise sine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand (in radians)
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the sine of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = sin(a)  # Compute sine
    ```

cutlass.cute.math.cos( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise cosine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand (in radians)
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the cosine of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = cos(a)  # Compute cosine
    ```

cutlass.cute.math.exp2( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise base-2 exponential of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing 2 raised to the power of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = exp2(a)  # Compute 2^x
    ```

cutlass.cute.math.log2( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise base-2 logarithm of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the base-2 logarithm of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = log2(a)  # Compute log base 2
    ```

cutlass.cute.math.tanh( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise hyperbolic tangent of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the hyperbolic tangent of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = tanh(a)  # Compute hyperbolic tangent
    ```

cutlass.cute.math.rsqrt( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise reciprocal square root of the input operand.

    Computes 1/√x element-wise.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the reciprocal square root of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = rsqrt(a)  # Compute 1/√x
    ```

cutlass.cute.math.sqrt( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *ftz: bool = False*, : *rounding: [RoundingMode](cute_math.md#cutlass.cute.math.RoundingMode "cutlass._mlir_helpers.math.RoundingMode") | None = None*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise square root of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the square root of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = sqrt(a)  # Compute square root
    ```

cutlass.cute.math.exp( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise exponential of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the exponential of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = exp(a)  # Compute exponential
    ```

cutlass.cute.math.log( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise natural logarithm of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the natural logarithm of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = log(a)  # Compute natural logarithm
    ```

cutlass.cute.math.log10( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise base-10 logarithm of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the base-10 logarithm of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = log10(a)  # Compute log base 10
    ```

cutlass.cute.math.tan( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise tangent of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand (in radians)
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the tangent of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = tan(a)  # Compute tangent
    ```

cutlass.cute.math.acos( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise arc cosine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the arc cosine of each element in input operand

    Return type:
    :   MathOperand

    Example:

    ```console
    y = acos(a)  # Compute arc cosine
    ```

cutlass.cute.math.asin( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise arc sine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the arc sine of each element in input operand

    Return type:
    :   MathOperand

    Example:

    ```console
    y = asin(a)  # Compute arc sine
    ```

cutlass.cute.math.atan( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise arc tangent of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the arc tangent of each element in input operand

    Return type:
    :   MathOperand

    Example:

    ```console
    y = atan(a)  # Compute arc tangent
    ```

cutlass.cute.math.atan2( : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise arc tangent of two tensors.

    Computes atan2(a, b) element-wise. The function atan2(a, b) is the angle in radians
    between the positive x-axis and the point given by the coordinates (b, a).

    Parameters:
    :   - **a** (*MathOperand*) – First input operand (y-coordinates)
        - **b** (*MathOperand*) – Second input operand (x-coordinates)
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the arc tangent of a/b element-wise

    Return type:
    :   MathOperand

    Example:

    ```console
    theta = atan2(y, x)  # Compute angles
    ```

cutlass.cute.math.sinh( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise hyperbolic sine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the hyperbolic sine of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = sinh(a)  # Compute hyperbolic sine
    ```

cutlass.cute.math.cosh( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise hyperbolic cosine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the hyperbolic cosine of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = cosh(a)  # Compute hyperbolic cosine
    ```

cutlass.cute.math.acosh( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise inverse hyperbolic cosine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand (must be >= 1.0)
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the inverse hyperbolic cosine of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = acosh(a)  # Compute inverse hyperbolic cosine
    ```

cutlass.cute.math.asinh( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise inverse hyperbolic sine of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the inverse hyperbolic sine of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = asinh(a)  # Compute inverse hyperbolic sine
    ```

cutlass.cute.math.atanh( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise inverse hyperbolic tangent of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand (must be in (-1, 1))
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the inverse hyperbolic tangent of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = atanh(a)  # Compute inverse hyperbolic tangent
    ```

cutlass.cute.math.erf( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise error function of the input operand.

    The error function is defined as:
    erf(x) = 2/√π ∫[0 to x] exp(-t²) dt

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the error function value for each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = erf(a)  # Compute error function
    ```

cutlass.cute.math.erfc( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise complementary error function of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the complementary error function of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = erfc(a)  # Compute complementary error function
    ```

cutlass.cute.math.pow( : *base: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *exponent: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise power of the input tensors.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand (base)
        - **b** (*MathOperand*) – Input operand (exponent)
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing a raised to the power b for each element

    Return type:
    :   MathOperand

    Example:

    ```console
    z = pow(a, b)  # Compute a^b
    ```

cutlass.cute.math.fpowi( : *base: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *exponent: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Float raised to integer power: base^exponent (exponent is integer).

cutlass.cute.math.ipowi( : *base: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *exponent: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Integer raised to integer power: base^exponent.

cutlass.cute.math.cbrt( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise cube root of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the cube root of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = cbrt(a)  # Compute cube root
    ```

cutlass.cute.math.expm1( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise exp(x) - 1 of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing exp(x) - 1 of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = expm1(a)  # Compute exp(x) - 1
    ```

cutlass.cute.math.log1p( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise log(1 + x) of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing log(1 + x) of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = log1p(a)  # Compute log(1 + x)
    ```

cutlass.cute.math.sincos( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → tuple
:   Combined sine and cosine: returns (sin(x), cos(x)).

    More efficient than separate sin() and cos() calls.

cutlass.cute.math.abs( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Absolute value (float or integer).

    Parameters:
    :   **ftz** – Flush denormals to zero (float only, uses nvvm.fabs)

cutlass.cute.math.absi( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Integer absolute value.

cutlass.cute.math.copysign( : *mag: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *sign: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise copysign, combining magnitude of a with sign of b.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand (magnitude source)
        - **b** (*MathOperand*) – Input operand (sign source)
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the copysign of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    z = copysign(a, b)  # Copy sign of b to magnitude of a
    ```

cutlass.cute.math.neg( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Floating-point negation.

cutlass.cute.math.ceil( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise ceiling of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the ceiling of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = ceil(a)  # Compute ceiling
    ```

cutlass.cute.math.floor( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise floor of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the largest integer less than or equal to each element in input operand

    Return type:
    :   MathOperand

    Example:

    ```console
    y = floor(a)  # Compute floor
    ```

cutlass.cute.math.round( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise round to nearest integer (ties away from zero) of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the rounded value of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = round(a)  # Round to nearest integer
    ```

cutlass.cute.math.roundeven( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise round to nearest integer (ties to even) of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the rounded value of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = roundeven(a)  # Round to nearest integer (ties to even)
    ```

cutlass.cute.math.trunc( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Compute element-wise truncation toward zero of the input operand.

    Parameters:
    :   - **a** (*MathOperand*) – Input operand
        - **fastmath** (*bool**,* *optional*) – Enable fast math optimizations, defaults to False
        - **loc** (*Optional**[**Location**]*) – Source location information, defaults to None
        - **ip** (*Optional**[**InsertionPoint**]*) – Insertion point for IR generation, defaults to None

    Returns:
    :   Result containing the truncated value of each element

    Return type:
    :   MathOperand

    Example:

    ```console
    y = trunc(a)  # Truncate toward zero
    ```

cutlass.cute.math.clamp( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *lo: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *hi: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *propagate\_nan: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Clamp value to range [lo, hi].

    Lowered unconditionally as `max(lo, min(x, hi))`. The math dialect’s
    `clampf` op currently has no LLVM translation interface registered, so
    a direct lowering fails at JIT time on scalar f16 inputs (and any other
    type). Composition via min/max picks up the right per-type lowering
    (`arith.minnumf` / `arith.maximumf` / `nvvm.fmin` / `nvvm.fmax`).

cutlass.cute.math.min( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *propagate\_nan: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Element-wise minimum.

    Parameters:
    :   - **propagate\_nan** – If True, NaN propagates (IEEE 754 minimum).
          If False, NaN is ignored (IEEE 754 minimumNumber).
        - **ftz** – Flush denormals to zero (float only, uses nvvm.fmin).

cutlass.cute.math.max( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *propagate\_nan: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Element-wise maximum.

    Parameters:
    :   - **propagate\_nan** – If True, NaN propagates (IEEE 754 maximum).
          If False, NaN is ignored (IEEE 754 maximumNumber).
        - **ftz** – Flush denormals to zero (float only, uses nvvm.fmax).

cutlass.cute.math.isnan( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Test if value is NaN. Returns i1.

cutlass.cute.math.isinf( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Test if value is infinite. Returns i1.

cutlass.cute.math.isfinite( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Test if value is finite. Returns i1.

cutlass.cute.math.isnormal( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Test if value is normal (not zero, subnormal, inf, or NaN). Returns i1.

cutlass.cute.math.fma( : *a: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *b: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *c: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *rounding: [RoundingMode](cute_math.md#cutlass.cute.math.RoundingMode "cutlass._mlir_helpers.math.RoundingMode") | None = None*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Fused multiply-add: a \* b + c.

cutlass.cute.math.add( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *rounding: [RoundingMode](cute_math.md#cutlass.cute.math.RoundingMode "cutlass._mlir_helpers.math.RoundingMode") | None = None*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Floating-point addition.

cutlass.cute.math.sub( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Floating-point subtraction.

    Note: NVVM sub uses a different intrinsic convention (rounding as an
    integer arg), so rounding/ftz are not supported here. Callers that
    need explicit rounding control on subtraction should emit the
    PTX-level `sub.<rn|rz|rm|rp>[.ftz].f32` intrinsic directly.

cutlass.cute.math.mul( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *rounding: [RoundingMode](cute_math.md#cutlass.cute.math.RoundingMode "cutlass._mlir_helpers.math.RoundingMode") | None = None*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Floating-point multiplication.

cutlass.cute.math.div( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *ftz: bool = False*, : *full: bool = False*, : *rounding: [RoundingMode](cute_math.md#cutlass.cute.math.RoundingMode "cutlass._mlir_helpers.math.RoundingMode") | None = None*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Floating-point division.

cutlass.cute.math.rem( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *y: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Floating-point remainder.

cutlass.cute.math.rcp( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *approx: bool = False*, : *rounding: [RoundingMode](cute_math.md#cutlass.cute.math.RoundingMode "cutlass._mlir_helpers.math.RoundingMode") | None = None*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Reciprocal (1/x).

cutlass.cute.math.absf( : *x: [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool*, : *fastmath: bool = False*, : *ftz: bool = False*, ) → [Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric") | [Vector](../basic_data_types.md#cutlass.Vector "cutlass.Vector") | float | int | bool
:   Absolute value (float or integer).

    Parameters:
    :   **ftz** – Flush denormals to zero (float only, uses nvvm.fabs)
