import { sql } from 'drizzle-orm';
import {
  index,
  jsonb,
  pgTable,
  timestamp,
  varchar,
  boolean,
  integer,
  text,
  unique,
  decimal,
  uuid,
  serial,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";

// Session storage table (required for Replit Auth)
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// User storage table (required for Replit Auth)
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  role: varchar("role").default("customer"), // customer, admin
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Tenants (Organizations) - Your main Voxcentia company
export const tenants = pgTable("tenants", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name").notNull(),
  status: varchar("status").default("active"), // active, inactive
  billingPlan: varchar("billing_plan").default("enterprise"), // free, basic, enterprise
  createdAt: timestamp("created_at").defaultNow(),
});

// Projects (Customer accounts under your tenant)
export const projects = pgTable("projects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  tenantId: varchar("tenant_id").references(() => tenants.id),
  userId: varchar("user_id").references(() => users.id), // Link to customer user
  name: varchar("name").notNull(),
  companyName: varchar("company_name"),
  contactEmail: varchar("contact_email"),
  isActive: boolean("is_active").default(true),
  billingPlan: varchar("billing_plan").default("basic"), // free, basic, pro, enterprise
  monthlyQuota: integer("monthly_quota").default(1000), // API calls per month
  createdAt: timestamp("created_at").defaultNow(),
});

// API Keys (existing table - matches current production structure)
export const apiKeys = pgTable("api_keys", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id),
  keyHash: text("key_hash").notNull(),
  label: text("label"),
  isActive: boolean("is_active").default(true),
  rateLimitPerMin: integer("rate_limit_per_min").default(1000), // matches actual DB column
  password: varchar("password"), // Simple password for customer login
  createdAt: timestamp("created_at").defaultNow(),
});

// API Usage Tracking
export const apiUsage = pgTable("api_usage", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id),
  apiKeyId: varchar("api_key_id").references(() => apiKeys.id),
  endpoint: varchar("endpoint").notNull(), // /v1/voice/analyze
  method: varchar("method").default("POST"),
  statusCode: integer("status_code"),
  responseTimeMs: integer("response_time_ms"),
  audioFileSize: integer("audio_file_size"), // bytes
  processingTimeMs: integer("processing_time_ms"),
  timestamp: timestamp("timestamp").defaultNow(),
});

// Monthly Usage Summary (for billing and analytics)
export const monthlyUsage = pgTable("monthly_usage", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id),
  year: integer("year").notNull(),
  month: integer("month").notNull(), // 1-12
  totalRequests: integer("total_requests").default(0),
  successfulRequests: integer("successful_requests").default(0),
  failedRequests: integer("failed_requests").default(0),
  totalAudioSeconds: integer("total_audio_seconds").default(0),
  avgResponseTimeMs: integer("avg_response_time_ms"),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  unique().on(table.projectId, table.year, table.month)
]);

// Voice Recordings (Vibeonix API)
export const recordings = pgTable("recordings", {
  id: serial("id").primaryKey(), // Sequential ID for recordingId
  uuid: uuid("uuid").default(sql`gen_random_uuid()`).unique(), // Internal UUID
  userId: varchar("user_id").notNull(), // Vibeonix user ID (from clients)
  projectId: uuid("project_id").references(() => projects.id),
  apiKeyId: uuid("api_key_id").references(() => apiKeys.id),
  filename: varchar("filename"),
  filepath: varchar("filepath"), // S3 or storage path
  contentHash: varchar("content_hash"), // Audio content hash for caching
  createdAt: timestamp("created_at").defaultNow(),
  
  // Vibeonix wellness scores (1-100 scale)
  moodScore: integer("mood_score"),
  anxietyScore: integer("anxiety_score"),
  stressScore: integer("stress_score"),
  happinessScore: integer("happiness_score"),
  lonelinessScore: integer("loneliness_score"),
  resilienceScore: integer("resilience_score"),
  energyScore: integer("energy_score"),
  
  // Additional metadata for compatibility
  audioMs: integer("audio_ms"), // Duration in milliseconds
  processingMs: integer("processing_ms"),
  
  // Store full analysis results as JSON (for advanced features)
  fullScores: jsonb("full_scores"), // Complete emo1-emo26 + char1-char94
});

// User Statistics and Trends (Vibeonix API)
export const userStatistics = pgTable("user_statistics", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull(), // Vibeonix user ID
  projectId: uuid("project_id").references(() => projects.id),
  
  // Recording counts and dates
  totalRecordings: integer("total_recordings").default(0),
  firstRecordingDate: timestamp("first_recording_date"),
  latestRecordingDate: timestamp("latest_recording_date"),
  
  // Wellness score averages (30-day rolling)
  moodAvg: integer("mood_avg"),
  anxietyAvg: integer("anxiety_avg"),
  stressAvg: integer("stress_avg"),
  happinessAvg: integer("happiness_avg"),
  lonelinessAvg: integer("loneliness_avg"),
  resilienceAvg: integer("resilience_avg"),
  energyAvg: integer("energy_avg"),
  
  // Trend indicators (-1=declining, 0=stable, 1=improving)
  moodTrend: integer("mood_trend").default(0),
  anxietyTrend: integer("anxiety_trend").default(0),
  stressTrend: integer("stress_trend").default(0),
  happinessTrend: integer("happiness_trend").default(0),
  lonelinessTrend: integer("loneliness_trend").default(0),
  resilienceTrend: integer("resilience_trend").default(0),
  energyTrend: integer("energy_trend").default(0),
  
  // Overall wellness indicator ID (1-10 scale)
  wellnessIndicatorId: integer("wellness_indicator_id").default(5),
  
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  unique().on(table.userId, table.projectId)
]);

// Wellness Indicators Descriptions (Vibeonix API)
export const indicators = pgTable("indicators", {
  id: integer("id").primaryKey(), // 1=mood, 2=anxiety, etc.
  name: varchar("name").notNull(),
  description: text("description"),
  tips: jsonb("tips"), // Array of strings
  positiveComponents: jsonb("positive_components"), // Array of strings  
  negativeComponents: jsonb("negative_components"), // Array of strings
});

// Relations
export const tenantsRelations = relations(tenants, ({ many }) => ({
  projects: many(projects),
}));

export const usersRelations = relations(users, ({ many }) => ({
  projects: many(projects),
}));

export const projectsRelations = relations(projects, ({ one, many }) => ({
  tenant: one(tenants, {
    fields: [projects.tenantId],
    references: [tenants.id],
  }),
  user: one(users, {
    fields: [projects.userId],
    references: [users.id],
  }),
  apiKeys: many(apiKeys),
  usage: many(apiUsage),
  monthlyUsage: many(monthlyUsage),
}));

export const apiKeysRelations = relations(apiKeys, ({ one, many }) => ({
  project: one(projects, {
    fields: [apiKeys.projectId],
    references: [projects.id],
  }),
  usage: many(apiUsage),
}));

export const apiUsageRelations = relations(apiUsage, ({ one }) => ({
  project: one(projects, {
    fields: [apiUsage.projectId],
    references: [projects.id],
  }),
  apiKey: one(apiKeys, {
    fields: [apiUsage.apiKeyId],
    references: [apiKeys.id],
  }),
}));

export const recordingsRelations = relations(recordings, ({ one }) => ({
  project: one(projects, {
    fields: [recordings.projectId],
    references: [projects.id],
  }),
  apiKey: one(apiKeys, {
    fields: [recordings.apiKeyId],
    references: [apiKeys.id],
  }),
}));

export const userStatisticsRelations = relations(userStatistics, ({ one }) => ({
  project: one(projects, {
    fields: [userStatistics.projectId],
    references: [projects.id],
  }),
}));

// Export types
export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;
export type Tenant = typeof tenants.$inferSelect;
export type Project = typeof projects.$inferSelect;
export type ApiKey = typeof apiKeys.$inferSelect;
export type ApiUsage = typeof apiUsage.$inferSelect;
export type MonthlyUsage = typeof monthlyUsage.$inferSelect;
export type Recording = typeof recordings.$inferSelect;
export type InsertRecording = typeof recordings.$inferInsert;
export type UserStatistics = typeof userStatistics.$inferSelect;
export type InsertUserStatistics = typeof userStatistics.$inferInsert;
export type Indicator = typeof indicators.$inferSelect;