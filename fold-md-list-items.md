# Fold markdown list items

search
```
(^(\s*)(((?:\d+\.)|\-|\+|\*)\s+(?:(?!\n\n)(?:.|\n))*))\n\n((.|\n)+?)(\n{0,2})(?=(^(\1|\2((?:\d+\.)|\-|\+|\*)\s+))|\z)
```

replace
```
\2<details>\n\n\2<summary> \3 </summary>\n\n\5\n\n\2</details>\7
```

---

```
1. Problema detalhado
em duas linhas

    Esta é a treta. foi feitos isso e aquilo
sdsadsadasdasadasd
    e tambem isso

2. a

    nao ficou tao bomasdasdasd

3. Teste

    Teste ai po

4. Suporte

    e isso

```