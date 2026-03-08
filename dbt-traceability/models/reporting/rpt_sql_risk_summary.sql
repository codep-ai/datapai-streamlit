{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'reporting', 'published', 'governance']
  )
}}

/*
  rpt_sql_risk_summary

  Aggregated SQL risk profile per datasource and user.
  Shows which users are generating high-risk SQL, against which datasources,
  and how often SQL is blocked vs approved vs executed.

  Powers:
    - Lightdash "SQL Risk" dashboard
    - Admin "SQL Risk Summary" console (spec G)
*/

with sql_events as (

    select *
    from {{ ref('stg_trace_events') }}
    where is_sql_event = true

),

summary as (

    select
        tenant_id,
        workspace_id,
        user_id,
        datasource_type,
        datasource_name,
        date_trunc('day', event_timestamp)     as event_date,

        -- Volume
        count(*)                               as total_sql_events,
        count_if(event_type = 'sql_generated') as sql_generated,
        count_if(event_type = 'sql_validated') as sql_validated,
        count_if(event_type = 'sql_executed')  as sql_executed,
        count_if(event_type = 'sql_blocked')   as sql_blocked,

        -- Risk ratio
        round(
            count_if(event_type = 'sql_blocked') * 100.0
            / nullif(count_if(event_type = 'sql_generated'), 0),
        1)                                     as block_rate_pct,

        -- Distinct SQL patterns
        count(distinct sql_hash)               as distinct_sql_hashes,

        -- Sensitivity / PII profile
        count_if(sensitivity_level = 'CRITICAL') as critical_sensitivity_count,
        count_if(sensitivity_level = 'HIGH')     as high_sensitivity_count,
        count_if(pii_detected = true)            as pii_event_count,

        -- Human corrections
        count_if(is_human_correction)          as human_corrections,

        -- Risk level (accounts for sensitivity and PII)
        case
            when count_if(event_type = 'sql_blocked') > 5    then 'HIGH'
            when count_if(event_type = 'sql_blocked') > 0    then 'MEDIUM'
            when count_if(sensitivity_level = 'CRITICAL') > 0 then 'MEDIUM'
            when count_if(pii_detected = true) > 0            then 'MEDIUM'
            when count_if(is_error)                   > 0    then 'LOW'
            else 'NONE'
        end                                    as risk_level

    from sql_events
    group by 1, 2, 3, 4, 5, 6

)

select * from summary
order by event_date desc, sql_blocked desc
