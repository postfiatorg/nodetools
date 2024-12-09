import pathlib
from typing import Optional, List
from loguru import logger

class SQLManager:
    """Manages SQL script loading and execution"""
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            base_path = pathlib.Path(__file__).parent.parent / 'sql'
        self.base_path = pathlib.Path(base_path)

    def load_query(self, category: str, name: str, module: Optional[str] = None) -> str:
        """Load SQL query from file
        
        Args:
            category: The category of SQL (e.g., 'init', 'queries')
            name: The name of the SQL file without extension
            module: Optional module name (e.g., 'discord')
            
        Returns:
            str: The contents of the SQL file
        """
        if module:
            file_path = self.base_path / module / f"{name}.sql"
        else:
            file_path = self.base_path / category / f"{name}.sql"
        
        try:
            return file_path.read_text()
        except FileNotFoundError:
            logger.error(f"SQL file not found: {file_path}")
            raise

    async def execute_script(self, db_manager, category: str, name: str, *args, module: Optional[str] = None):
        """Execute a SQL script using the database manager
        
        Args:
            db_manager: Database manager instance
            category: The category of SQL (e.g., 'init', 'queries')
            name: The name of the SQL file without extension
            *args: Arguments to pass to the query
            module: Optional module name (e.g., 'discord')
        """
        query = self.load_query(category, name, module)
        return await db_manager.execute(query, *args)
    
    def initialize_module(self, db_manager, module: str):
        """Initialize a specific module's database objects
        
        Args:
            db_manager: Database manager instance
            module: The name of the module to initialize (e.g., 'discord')
        """
        logger.info(f"Initializing database objects for module: {module}")
        
        # Order matters: tables first, then indices, then views
        initialization_files = [
            ('create_tables', 'Creating tables'),
            ('create_indices', 'Creating indices'),
            ('create_views', 'Creating views')
        ]
        
        for file_name, description in initialization_files:
            try:
                query = self.load_query(file_name, module=module)
                logger.info(f"{description} for module {module}")
                db_manager.execute(query)
            except Exception as e:
                logger.error(f"Error {description.lower()} for module {module}: {e}")
                raise

    def initialize_all(self, db_manager):
        """Initialize all database objects including module-specific ones
        
        Args:
            db_manager: Database manager instance
        """
        logger.info("Starting complete database initialization")
        
        # Initialize core tables first
        try:
            # Core initialization
            for category in ['create_tables', 'create_indices', 'create_views']:
                query = self.load_query('init', category)
                logger.info(f"Executing core {category}")
                db_manager.execute(query)
                
            # Find and initialize all modules
            modules = [d for d in self.base_path.iterdir() if d.is_dir() and d.name != 'init']
            
            for module_path in modules:
                self.initialize_module(db_manager, module_path.name)
                
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            raise