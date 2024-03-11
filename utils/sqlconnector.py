from sqlalchemy import create_engine

host = "plotly-customer-success.cloud.databricks.com"
token = "dapic65e825bf500366e1fb87ab797d3e4fd"
path = "/sql/1.0/warehouses/07bdd5688d399f3d"
engine_url = f"databricks://token:{token}@{host}/?http_path={path}&catalog=main&schema=information_schema"
engine = create_engine(engine_url)


from sqlalchemy import text
import pandas as pd



def execute_sql_and_return_df(engine, stmt):
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)
    return df


# def execute_sql_and_return_final_df(engine, create_schema_stmt, pre_stmt, final_stmt):
#     with engine.connect() as conn:
#         with conn.begin():
#             # Convert SQL statements to Text objects
#             conn.execute(text(create_schema_stmt))
#             conn.execute(text(pre_stmt))

#     # Execute the final statement and return DataFrame
#     df = pd.read_sql(final_stmt, engine)
#     return df

# # SQL statements
# create_schema_stmt = "CREATE SCHEMA IF NOT EXISTS my_schema"