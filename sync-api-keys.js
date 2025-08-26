/**
 * Sync API keys from Customer Management Platform to Supabase
 * This ensures API keys work with the Railway Voice Analysis API
 */

const { createClient } = require('@supabase/supabase-js');
const crypto = require('crypto');

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function syncApiKeys() {
  try {
    console.log('🔄 Starting API key sync...');
    
    // Load customer API keys from environment variables (secure approach)
    const knownApiKeys = [];
    
    // Parse API keys from environment variables
    const apiKeyPrefix = 'CUSTOMER_API_KEY_';
    const labelPrefix = 'CUSTOMER_LABEL_';
    
    for (let i = 1; i <= 5; i++) {
      const apiKey = process.env[`${apiKeyPrefix}${i}`];
      const label = process.env[`${labelPrefix}${i}`];
      
      if (apiKey && label) {
        knownApiKeys.push({
          raw_key: apiKey,
          label: label,
          rate_limit: 1000
        });
      }
    }
    
    if (knownApiKeys.length === 0) {
      console.error('❌ No customer API keys found in environment variables');
      console.error('   Set CUSTOMER_API_KEY_1, CUSTOMER_LABEL_1, etc. via Replit secrets');
      process.exit(1);
    }
    
    console.log(`📋 Syncing ${knownApiKeys.length} customer API keys`);
    
    // Check existing keys in Supabase
    const { data: existingKeys, error: fetchError } = await supabase
      .from('api_keys')
      .select('key_hash');
    
    if (fetchError) {
      console.error('❌ Error fetching existing keys from Supabase:', fetchError);
      return;
    }
    
    const existingHashes = new Set(existingKeys?.map(k => k.key_hash) || []);
    console.log(`📋 Found ${existingHashes.size} existing keys in Supabase`);
    
    // Sync missing keys
    let syncedCount = 0;
    let skippedCount = 0;
    
    for (const keyData of knownApiKeys) {
      // Hash the raw key
      const keyHash = crypto.createHash('sha256').update(keyData.raw_key).digest('hex');
      
      if (existingHashes.has(keyHash)) {
        console.log(`⏭️  Skipping existing key: ${keyData.label} (${keyHash.substring(0, 8)}...)`);
        skippedCount++;
        continue;
      }
      
      // Generate UUID for the key
      const keyId = crypto.randomUUID();
      
      // Insert key into Supabase
      const { error: insertError } = await supabase
        .from('api_keys')
        .insert({
          id: keyId,
          key_hash: keyHash,
          label: keyData.label,
          is_active: true,
          rate_limit: keyData.rate_limit,
          created_at: new Date().toISOString()
        });
      
      if (insertError) {
        console.error(`❌ Error syncing key ${keyData.label}:`, insertError);
      } else {
        console.log(`✅ Synced key: ${keyData.label} (${keyHash.substring(0, 8)}...)`);
        syncedCount++;
      }
    }
    
    console.log('\n🎯 Sync Complete!');
    console.log(`✅ Synced: ${syncedCount} keys`);
    console.log(`⏭️  Skipped: ${skippedCount} keys (already existed)`);
    console.log(`📊 Total: ${knownApiKeys.length} customer API keys`);
    
    if (syncedCount > 0) {
      console.log('\n🚀 Your customers can now use their API keys with the Railway Voice Analysis API!');
      console.log('Test with: https://voice-analysis-api-production.up.railway.app/v1/voice/analyze');
    }
    
  } catch (error) {
    console.error('❌ Sync failed:', error);
  }
}

// Run if called directly
if (require.main === module) {
  syncApiKeys().then(() => process.exit(0));
}

module.exports = { syncApiKeys };