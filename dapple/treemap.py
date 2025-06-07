from xml.etree import ElementTree as ET
from typing import TypeVar, Callable, Optional, Tuple, Union

# Type variable for state
S = TypeVar('S')


def treemap(
    tree: ET.ElementTree,
    func: Callable[[ET.Element, S], Tuple[Optional[ET.Element], S]],
    state: S
) -> Tuple[Optional[ET.ElementTree], S]:
    """
    Traverse and optionally rewrite an ElementTree according to a user-provided function.
    
    The function traverses the tree in a depth-first manner, applying the provided
    function to each element. The function can modify elements and maintain state
    as it traverses.
    
    Args:
        tree: The ElementTree to traverse/rewrite
        func: A function that takes an Element and state, returns a tuple of
              (Optional[Element], state). If the Element is None, the node is
              removed from the tree.
        state: Initial state that will be passed through and potentially modified
               during traversal
    
    Returns:
        A tuple of (Optional[ElementTree], final_state). The ElementTree will be
        None if the root element was mapped to None.
    
    Example:
        >>> # Count elements while transforming tags to uppercase
        >>> def transform_and_count(elem, counts):
        ...     counts[elem.tag] = counts.get(elem.tag, 0) + 1
        ...     new_elem = ET.Element(elem.tag.upper())
        ...     new_elem.text = elem.text
        ...     new_elem.attrib = elem.attrib.copy()
        ...     return new_elem, counts
        >>> 
        >>> tree = ET.ElementTree(ET.fromstring('<root><child>text</child></root>'))
        >>> new_tree, counts = treemap(tree, transform_and_count, {})
        >>> print(counts)  # {'root': 1, 'child': 1}
    """
    root = tree.getroot()
    if root is None:
        return None, state
    
    # Process the tree starting from the root
    new_root, final_state = _process_element(root, func, state)
    
    if new_root is None:
        return None, final_state
    
    # Create a new tree with the processed root
    new_tree = ET.ElementTree(new_root)
    return new_tree, final_state


def _process_element(
    element: ET.Element,
    func: Callable[[ET.Element, S], Tuple[Optional[ET.Element], S]],
    state: S
) -> Tuple[Optional[ET.Element], S]:
    """
    Process an element and its children recursively.
    
    Args:
        element: The element to process
        func: The transformation function
        state: Current state
    
    Returns:
        Tuple of (transformed element or None, updated state)
    """
    # Apply the function to this element
    transformed, new_state = func(element, state)
    
    # If the function returns None, remove this element
    if transformed is None:
        return None, new_state
    
    # Otherwise, we have a transformed element
    # Now process its children
    current_state = new_state
    for child in element:
        new_child, current_state = _process_element(child, func, current_state)
        if new_child is not None:
            transformed.append(new_child)
    
    return transformed, current_state


def treemap_pure(
    tree: ET.ElementTree,
    func: Callable[[ET.Element], Optional[ET.Element]]
) -> Optional[ET.ElementTree]:
    """
    A simpler version of treemap that doesn't maintain state.
    
    Args:
        tree: The ElementTree to traverse/rewrite
        func: A function that takes an Element and returns an Optional[Element].
              Return None to remove the element from the tree.
    
    Returns:
        The transformed ElementTree, or None if the root was mapped to None
    """
    # Wrapper to adapt the pure function to the stateful interface
    def wrapper(elem: ET.Element, state: None) -> Tuple[Optional[ET.Element], None]:
        return func(elem), None
    
    result_tree, _ = treemap(tree, wrapper, None)
    return result_tree


def treemap_traverse(
    tree: ET.ElementTree,
    func: Callable[[ET.Element, S], S],
    state: S
) -> S:
    """
    Traverse an ElementTree without rewriting it, only updating state.
    
    This is useful when you want to collect information from the tree
    without modifying it.
    
    Args:
        tree: The ElementTree to traverse
        func: A function that takes an Element and state, returns updated state
        state: Initial state that will be passed through and potentially modified
               during traversal
    
    Returns:
        The final state after traversing all elements
    """
    root = tree.getroot()
    if root is None:
        return state
    
    return _traverse_element(root, func, state)


def _traverse_element(
    element: ET.Element,
    func: Callable[[ET.Element, S], S],
    state: S
) -> S:
    """
    Helper function to traverse an element and its children without modification.
    
    Args:
        element: The element to traverse
        func: The state update function
        state: Current state
    
    Returns:
        Updated state after traversing this element and its children
    """
    # Update state for this element
    current_state = func(element, state)
    
    # Traverse children
    for child in element:
        current_state = _traverse_element(child, func, current_state)
    
    return current_state


def copy_element(element: ET.Element, include_children: bool = False) -> ET.Element:
    """
    Create a shallow or deep copy of an XML element.
    
    Args:
        element: The element to copy
        include_children: If True, recursively copy all children
    
    Returns:
        A new element with copied attributes, text, and optionally children
    """
    # Create new element with same tag and attributes
    new_elem = ET.Element(element.tag, element.attrib)
    new_elem.text = element.text
    new_elem.tail = element.tail
    
    if include_children:
        for child in element:
            new_elem.append(copy_element(child, include_children=True))
    
    return new_elem