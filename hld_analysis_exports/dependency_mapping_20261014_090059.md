# Dependency Mapping

*Generated: 20261014_090059*

## Metadata
- **query**: Identify the explicit dependencies and interactions between software components, interfaces, RTE, BSW, and other architecture entities.

## Output

DEPENDENCIES:

1. SOURCE:
   - Name: RTE
   - Depends on / interacts with: AUTOSAR Software Components
   - Relationship: Provides communication services to the application software
   - Page: [Page 19]

2. SOURCE:
   - Name: BSW
   - Depends on / interacts with: AUTOSAR Software Components
   - Relationship: Enables interlaced execution sequences of Runnable Entities and BSW Main functions
   - Page: [Page 135]

3. SOURCE:
   - Name: RTE
   - Depends on / interacts with: ECU Abstraction Layer
   - Relationship: The RTE is a layer providing communication services to the application software (ECU Abstraction Layer)
   - Page: [Page 19]

4. SOURCE:
   - Name: BSW
   - Depends on / interacts with: ECU Abstraction Layer
   - Relationship: Restrict the usage of OS functionality
   - Page: [Page 135]

5. SOURCE:
   - Name: RTE
   - Depends on / interacts with: OS
   - Relationship: Only the BSW Scheduler and the RTE shall use OS objects or OS services
   - Page: [Page 135]

6. SOURCE:
   - Name: BSW
   - Depends on / interacts with: OS
   - Relationship: Restrict the usage of OS functionality
   - Page: [Page 135]

7. SOURCE:
   - Name: ECU Abstraction Layer
   - Depends on / interacts with: RTE
   - Relationship: The RTE is a layer providing communication services to the application software (ECU Abstraction Layer)
   - Page: [Page 19]

8. SOURCE:
   - Name: ECU Abstraction Layer
   - Depends on / interacts with: BSW
   - Relationship: Restrict the usage of OS functionality
   - Page: [Page 135]

No explicit dependencies found in the provided context.
