```bash

mattergen-app/
├── backend
│   ├── core
│   │   ├── logging_config.py
│   │   ├── middleware_config.py
│   │   └── settings.py
│   ├── models
│   │   ├── download.py
│   │   ├── generate.py
│   │   └── retrieve.py
│   ├── routes
│   │   ├── download.py
│   │   ├── generate_lattice.py
│   │   └── retrieval.py
│   ├── services
│   │   ├── download_service.py
│   │   ├── generate_service.py
│   │   ├── retrieval_service.py
│   │   ├── store_service.py
│   │   └── utils.py
│   ├── cleanup.sh
│   ├── database.py
│   ├── deps.py
│   └── main.py
├── frontend
├── .env
├── .env.example
├── README.md
└── requirements.txt

```

## Some notes on Tailwind CSS (v4)
- [Tailwind CSS upgrade guide from v3 to v4](https://tailwindcss.com/docs/upgrade-guide#changes-from-v3)

- [Tailwind CSS with Vite](https://tailwindcss.com/docs/installation/using-vite)

To install node package for the CLI interface, run:
```bash
npm install tailwindcss @tailwindcss/vite
```