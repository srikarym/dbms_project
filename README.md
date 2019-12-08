# dbms
A miniature relational database with order. Course project for CSCI-GA.2433-001 : Database systems

Student details: 
- Manikanta Srikar Yellapragada : msy290
- Sudharshann D. : sd3770

# Requirements
- numpy
- BTrees

Btrees implementation taken from [here](https://github.com/zopefoundation/BTrees).
# Usage

## Inputs
- `queries.txt` which contains the queries to execute
- `sales1.txt` and `sales2.txt` contain the tables


Execute the program as
```
python3 main.py
```
Execution time of each query is displayed to the Standard output
## Outputs
- `netid_AllOperations` has the output of all operations 
- `<TableName>.txt` has the output when Outputtofile is used as a query
