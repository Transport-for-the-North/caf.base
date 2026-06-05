.. _`def-segments`:

Segments
========

All the segments available within :mod:`caf.base`, accessible through the
:class:`~caf.base.segments.SegmentsSuper` enum. Segments are defined as YAML
files within `{{ relative_folder }} <{{ url }}/{{ relative_folder }}>`_.

{% for name, segment in segments.items() -%}

.. _`def-{{ segment.name }}`:

{{ name|replace("_", " ")|title }}
{{ "-" * name|length }}

- Name: {{ segment.name }}{% if segment.alias %} ({{ segment.alias }}) {%- endif %}
- Enum: :attr:`~caf.base.segments.SegmentsSuper.{{ name|upper }}`

{%- if segment.exclusions %}
- Exclusions with: {% for value in segment.exclusions -%}
:ref:`def-{{ value.other_name }}`{%- if not loop.last %}, {% endif %}
{%- endfor -%}
{%- endif %}

{%- if segment.lookups %}
- Lookups with: {% for value in segment.lookups -%}
:ref:`def-{{ value.other_name }}`{%- if not loop.last %}, {% endif %}
{%- endfor -%}
{%- endif %}

.. csv-table::
    :header: "ID", "Value"{% if segment.values_aliases %}, "Alias"{% endif %}
    :stub-columns: 1

    {% for id, name in segment.values.items() %}
    {{ id }}, "{{ name }}"{% if segment.values_aliases %}, "{{ segment.values_aliases[id] }}"{% endif %}
    {%- endfor %}

{% endfor %}
